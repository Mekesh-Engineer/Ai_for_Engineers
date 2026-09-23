import json
import pytest

def test_project_crud(client):
    # 1. Create project
    res = client.post("/api/projects/", json={"name": "EV Battery Pack Design", "description": "Thermal and electrical sizing"})
    assert res.status_code == 201
    data = res.get_json()
    proj_id = data["id"]
    assert data["name"] == "EV Battery Pack Design"

    # 2. List projects
    res_list = client.get("/api/projects/")
    assert res_list.status_code == 200
    assert any(p["id"] == proj_id for p in res_list.get_json())

    # 3. Get detail
    res_detail = client.get(f"/api/projects/{proj_id}")
    assert res_detail.status_code == 200
    assert res_detail.get_json()["id"] == proj_id

    # 4. Update
    res_update = client.put(f"/api/projects/{proj_id}", json={"notes": "Updated cell chemistry to LFP"})
    assert res_update.status_code == 200
    assert res_update.get_json()["notes"] == "Updated cell chemistry to LFP"

    # 5. Delete
    res_del = client.delete(f"/api/projects/{proj_id}")
    assert res_del.status_code == 200
