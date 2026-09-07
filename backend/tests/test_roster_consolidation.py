import numpy as np
import pytest

from app.cv.roster_consolidation import (
    TrackFragment,
    hist_distance,
    find_candidates,
    consolidate,
    assign_roster_numbers,
    appearance_descriptor_from_crop,
)

FPS = 30.0
PX_TO_M = 0.05  # arbitrary simple scale for these fixtures


def test_hist_distance_identical_is_zero_and_disjoint_is_one():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert hist_distance(a, a) == pytest.approx(0.0, abs=1e-9)
    assert hist_distance(a, b) == pytest.approx(1.0, abs=1e-9)
    assert hist_distance(None, a) == float("inf")


def test_confident_pair_gets_auto_merged():
    # Same jersey-color signature, a short plausible gap - clearly one player
    # lost and re-spawned as a new track ID.
    a = TrackFragment(1, "HOME", [(10, (100.0, 100.0)), (50, (110.0, 100.0))], np.array([1.0, 0.0]))
    b = TrackFragment(2, "HOME", [(200, (115.0, 100.0)), (250, (120.0, 100.0))], np.array([1.0, 0.0]))

    result = consolidate([a, b], FPS, PX_TO_M)

    assert sorted(result.groups, key=len) == [sorted([1, 2])]
    assert result.ambiguous == []


def test_overlapping_fragments_never_merge_even_with_identical_appearance():
    # Two fragments active at the same time can't be the same real person,
    # no matter how similar they look - this must win over appearance.
    a = TrackFragment(1, "HOME", [(10, (100.0, 100.0)), (50, (100.0, 100.0))], np.array([1.0, 0.0]))
    b = TrackFragment(2, "HOME", [(30, (100.0, 100.0)), (80, (100.0, 100.0))], np.array([1.0, 0.0]))

    result = consolidate([a, b], FPS, PX_TO_M)

    assert sorted(result.groups) == [[1], [2]]
    assert result.ambiguous == []


def test_implausible_speed_gap_is_rejected_outright():
    # A gap that would require an impossible sprint speed can't be the same
    # person, however identical the appearance looks - not merged, and not
    # even surfaced as "ambiguous" (it's not a plausible candidate at all).
    a = TrackFragment(1, "HOME", [(10, (0.0, 0.0)), (20, (0.0, 0.0))], np.array([1.0, 0.0]))
    b = TrackFragment(2, "HOME", [(21, (100000.0, 0.0)), (30, (100000.0, 0.0))], np.array([1.0, 0.0]))

    candidates = find_candidates([a, b], FPS, PX_TO_M)
    result = consolidate([a, b], FPS, PX_TO_M)

    assert candidates == []
    assert sorted(result.groups) == [[1], [2]]
    assert result.ambiguous == []


def test_different_teams_never_merge():
    a = TrackFragment(1, "HOME", [(10, (100.0, 100.0)), (50, (100.0, 100.0))], np.array([1.0, 0.0]))
    b = TrackFragment(2, "AWAY", [(200, (100.0, 100.0)), (250, (100.0, 100.0))], np.array([1.0, 0.0]))

    result = consolidate([a, b], FPS, PX_TO_M)

    assert sorted(result.groups) == [[1], [2]]


def test_ambiguous_pair_is_surfaced_not_auto_merged():
    # Distance ~0.39 - plausible but well short of the confident threshold.
    a = TrackFragment(1, "HOME", [(10, (100.0, 100.0)), (50, (100.0, 100.0))], np.array([0.95, 0.05]))
    b = TrackFragment(2, "HOME", [(200, (105.0, 100.0)), (250, (110.0, 100.0))], np.array([0.5, 0.5]))

    result = consolidate([a, b], FPS, PX_TO_M)

    assert sorted(result.groups) == [[1], [2]]  # left unmerged
    assert len(result.ambiguous) == 1
    assert {result.ambiguous[0].a.track_id, result.ambiguous[0].b.track_id} == {1, 2}
    assert 0.3 <= result.ambiguous[0].distance < 0.45


def test_transitive_merge_through_a_third_fragment_is_blocked_by_overlap():
    # Regression test for the real risk this module's docstrings call out:
    # two teammates in identical kits can each look confidently similar to a
    # third, unrelated fragment (B) by appearance alone, even though A and C
    # were themselves active at the same time and so provably CANNOT be the
    # same person. A confident A-B link and a confident C-B link must not be
    # allowed to transitively merge A and C together.
    a = TrackFragment(1, "HOME", [(10, (100.0, 100.0)), (50, (100.0, 100.0))], np.array([1.0, 0.0]))
    c = TrackFragment(3, "HOME", [(30, (100.0, 100.0)), (70, (100.0, 100.0))], np.array([1.0, 0.0]))
    b = TrackFragment(2, "HOME", [(200, (100.0, 100.0)), (250, (100.0, 100.0))], np.array([1.0, 0.0]))

    result = consolidate([a, c, b], FPS, PX_TO_M)

    groups_by_member = {tid: frozenset(g) for g in result.groups for tid in g}
    # A and B merged (a real, valid link)...
    assert groups_by_member[1] == groups_by_member[2]
    # ...but C must NOT have been pulled into that same group, since A and C
    # overlapped in time and can't possibly be the same person.
    assert groups_by_member[3] != groups_by_member[1]
    assert 3 not in groups_by_member[1]


def test_assign_roster_numbers_prioritizes_longer_lived_groups():
    groups = [[1, 2], [3], [4], [5]]

    assignment, unresolved = assign_roster_numbers(groups, roster_size=3)

    # The 2-member group (more evidence it's a real, distinct player) is kept.
    assert assignment[1] == assignment[2]
    assert assignment[3] != assignment[4]
    assert set(assignment.values()) == {1, 2, 3}
    assert unresolved == [[5]]


def test_appearance_descriptor_from_crop_distinguishes_colors_and_matches_itself():
    # Blue vs red - two genuinely different HUES. Black vs white would NOT
    # work here: this descriptor uses hue+saturation only (the same
    # lighting-invariant scheme TeamAssignmentMixin uses for team colors -
    # see its docstring), and pure black/white are both achromatic
    # (near-zero saturation regardless of value), so they're indistinguishable
    # by design, not a bug.
    blue_crop = np.full((200, 100, 3), (200, 60, 20), dtype=np.uint8)  # BGR blue-ish
    red_crop = np.full((200, 100, 3), (20, 30, 200), dtype=np.uint8)  # BGR red-ish

    blue_a = appearance_descriptor_from_crop(blue_crop)
    blue_b = appearance_descriptor_from_crop(blue_crop.copy())
    red = appearance_descriptor_from_crop(red_crop)

    assert blue_a is not None and red is not None
    assert hist_distance(blue_a, blue_b) == pytest.approx(0.0, abs=1e-6)
    assert hist_distance(blue_a, red) > 0.3
    assert appearance_descriptor_from_crop(None) is None
    assert appearance_descriptor_from_crop(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_assign_roster_numbers_handles_fewer_groups_than_roster_size():
    groups = [[1], [2]]

    assignment, unresolved = assign_roster_numbers(groups, roster_size=8)

    assert len(set(assignment.values())) == 2
    assert unresolved == []
