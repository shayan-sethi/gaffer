import numpy as np
import pytest

from app.cv.player_gallery import PlayerGallery


def test_identify_returns_the_closest_confident_reference():
    gallery = PlayerGallery()
    gallery.add_reference("grey_17", np.array([1.0, 0.0, 0.0]))
    gallery.add_reference("white_7", np.array([0.0, 1.0, 0.0]))

    assert gallery.identify(np.array([0.98, 0.02, 0.0])) == "grey_17"
    assert gallery.identify(np.array([0.05, 0.95, 0.0])) == "white_7"


def test_identify_returns_none_when_nothing_is_close_enough():
    gallery = PlayerGallery()
    gallery.add_reference("grey_17", np.array([1.0, 0.0, 0.0]))

    # Orthogonal - distance 1.0, nowhere near DEFAULT_MAX_DISTANCE.
    assert gallery.identify(np.array([0.0, 1.0, 0.0])) is None


def test_identify_handles_empty_gallery_and_none_descriptor():
    gallery = PlayerGallery()
    assert gallery.identify(np.array([1.0, 0.0])) is None
    assert gallery.identify(None) is None

    gallery.add_reference("grey_17", np.array([1.0, 0.0]))
    assert gallery.identify(None) is None


def _torso_legs_crop(torso_bgr, legs_bgr, seed):
    # A flat solid-color crop is a degenerate case for a hue/sat histogram -
    # with zero pixel spread, the whole histogram is a single-bin spike, so
    # even a tiny hue shift that crosses a bin boundary reads as a huge
    # distance. A real photo always has texture/noise spreading energy across
    # neighboring bins, which is what makes similar-but-not-identical colors
    # register as similar rather than as a coin flip on bin edges - add a
    # little synthetic noise so these fixtures behave like a real crop would.
    rng = np.random.RandomState(seed)
    crop = np.zeros((200, 100, 3), dtype=np.uint8)
    torso_noise = rng.randint(-8, 9, size=(110, 100, 3))
    legs_noise = rng.randint(-8, 9, size=(90, 100, 3))
    crop[:110] = np.clip(np.array(torso_bgr) + torso_noise, 0, 255).astype(np.uint8)
    crop[110:] = np.clip(np.array(legs_bgr) + legs_noise, 0, 255).astype(np.uint8)
    return crop


def test_build_reference_averages_multiple_crops():
    gallery = PlayerGallery()
    # Two sightings of the "same" player (blue shirt, black shorts) under
    # slightly different lighting - the averaged reference should still
    # match a fresh sighting of the same kit, and reject a different kit.
    crop_a = _torso_legs_crop((200, 60, 20), (20, 20, 20), seed=1)
    crop_b = _torso_legs_crop((190, 70, 25), (25, 25, 25), seed=2)
    gallery.build_reference("blue_player", [crop_a, crop_b])

    assert "blue_player" in gallery.references

    same_kit_probe = _torso_legs_crop((195, 65, 22), (22, 22, 22), seed=3)
    different_kit_probe = _torso_legs_crop((20, 30, 200), (200, 200, 200), seed=4)  # red shirt, white shorts

    assert gallery.identify_crop(same_kit_probe) == "blue_player"
    assert gallery.identify_crop(different_kit_probe) is None


def test_build_reference_with_no_valid_crops_adds_nothing():
    gallery = PlayerGallery()
    gallery.build_reference("nobody", [None, np.zeros((0, 0, 3), dtype=np.uint8)])
    assert "nobody" not in gallery.references
