def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Gaffer Sports Intelligence"
    assert data["status"] == "online"


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_camera_registration_and_heartbeat(client):
    # 1. Register
    reg_payload = {
        "serial_number": "TEST-CAM-UNIT-001",
        "turf_name": "San Siro Pitch 3",
        "turf_id": "turf_milan_03",
        "ip_address": "192.168.1.105",
    }
    res = client.post("/api/cameras/register", json=reg_payload)
    assert res.status_code == 201
    cam_data = res.json()
    assert cam_data["serial_number"] == "TEST-CAM-UNIT-001"
    assert "api_key" in cam_data
    camera_id = cam_data["id"]

    # 2. Heartbeat
    hb_res = client.post(f"/api/cameras/{camera_id}/heartbeat", json={"status": "ONLINE"})
    assert hb_res.status_code == 200
    assert hb_res.json()["status"] == "ONLINE"


def test_match_lifecycle(client):
    # 1. Register Camera
    reg_res = client.post("/api/cameras/register", json={
        "serial_number": "TEST-CAM-MATCH-002",
        "turf_name": "Camp Nou Pitch 1",
    })
    camera_id = reg_res.json()["id"]

    # 2. Start Match
    start_res = client.post(f"/api/cameras/{camera_id}/start_match", json={
        "title": "Champions League Final 5v5",
        "expected_duration_sec": 300,
    })
    assert start_res.status_code == 200
    match_data = start_res.json()
    match_id = match_data["id"]
    assert match_data["status"] == "RECORDING"

    # 3. Check Next Match Poll
    poll_res = client.get(f"/api/cameras/{camera_id}/next_match")
    assert poll_res.status_code == 200
    assert poll_res.json()["has_active_match"] is True

    # 4. Fetch Match Details
    detail_res = client.get(f"/api/matches/{match_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == match_id

    # 5. Fetch Match Stats
    stats_res = client.get(f"/api/matches/{match_id}/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert "home_score" in stats_data
    assert "player_stats" in stats_data


def test_camera_calibration(client):
    reg_res = client.post("/api/cameras/register", json={
        "serial_number": "TEST-CAM-CALIB-003",
        "turf_name": "Wembley Pitch 5",
    })
    camera_id = reg_res.json()["id"]
    assert reg_res.json()["calibration_points"] is None

    # A valid, non-degenerate quad is accepted and persisted.
    points = [[400, 100], [880, 100], [1230, 650], [50, 650]]
    set_res = client.put(f"/api/cameras/{camera_id}/calibration", json={"points": points})
    assert set_res.status_code == 200
    assert set_res.json()["calibration_points"] == points

    get_res = client.get(f"/api/cameras/{camera_id}")
    assert get_res.json()["calibration_points"] == points


def test_camera_calibration_rejects_degenerate_points(client):
    reg_res = client.post("/api/cameras/register", json={
        "serial_number": "TEST-CAM-CALIB-004",
        "turf_name": "Wembley Pitch 6",
    })
    camera_id = reg_res.json()["id"]

    # All 4 points share the same y - collinear, not a real quad.
    collinear = [[0, 100], [100, 100], [200, 100], [300, 100]]
    res = client.put(f"/api/cameras/{camera_id}/calibration", json={"points": collinear})
    assert res.status_code == 422


def test_camera_calibration_requires_exactly_four_points(client):
    reg_res = client.post("/api/cameras/register", json={
        "serial_number": "TEST-CAM-CALIB-005",
        "turf_name": "Wembley Pitch 7",
    })
    camera_id = reg_res.json()["id"]

    res = client.put(f"/api/cameras/{camera_id}/calibration", json={"points": [[0, 0], [1, 1], [2, 2]]})
    assert res.status_code == 422


def test_camera_calibration_unknown_camera_404(client):
    res = client.put("/api/cameras/does-not-exist/calibration", json={
        "points": [[400, 100], [880, 100], [1230, 650], [50, 650]]
    })
    assert res.status_code == 404
