import json 
def save_results(ocp_dist, ocp_vel, ocp_nocol, scene):
    data = {
        "xs_dist": [array.tolist() for array in ocp_dist.xs.tolist()],
        "us_dist": [array.tolist() for array in ocp_dist.us.tolist()],
        "xs_vel": [array.tolist() for array in ocp_vel.xs.tolist()],
        "us_vel": [array.tolist() for array in ocp_vel.us.tolist()],
        "xs_nocol": [array.tolist() for array in ocp_nocol.xs.tolist()],
        "us_nocol": [array.tolist() for array in ocp_nocol.us.tolist()],
    }
    with open("results/" +  str(scene) + ".json", "w") as json_file:
        json.dump(data, json_file, indent=6)
    print("saved in", "results/" +  str(scene) + ".json",)