import numpy as np
GROUP_DISTANCE_THRESHOLD = 2.1
def calulate_num_groups(axles, group_distance_threshold=GROUP_DISTANCE_THRESHOLD):
    num_groups = 1+ sum(np.array(axles)>=group_distance_threshold)
    return num_groups

def vehicle_class_from_axles(axles):
    """

    Args:
        axles: list of axle distance. Example: [5.5,3.5] representing 3-axle vehicle. Distance between ax-1 and ax-2 is 5.5m,
        distance between ax-2 and ax-3 is 3.5m

    Returns:

    """
    if len(axles) < 1:
        return -1

    total_length = sum(axles)
    num_groups = calulate_num_groups(axles)
    is_short = total_length<=5.5
    is_medium = 5.5 <= total_length<=14.5
    is_long = 11.5<=total_length<=19.0
    is_medium_comb = 17.5 <= total_length <=36.5
    is_long_comb = total_length>33

    # class 1:
    num_axles = len(axles) + 1
    if is_short:
        if num_axles == 2 and axles[0] <= 3.2:
            return 1
        if (3 <= num_axles <=5) and (2.1 <= axles[0] <= 3.2) and (axles[1] >= 2.1):
            return 2

    if is_medium:
        if num_axles == 2 and num_groups ==2 and (axles[0] > 3.2):
            return 3
        if num_axles == 3 and num_groups == 2:
            return 4
        if num_groups == 2 and num_axles > 3:
            return 5

    if is_long:
        if num_axles == 3 and num_groups == 3 and (axles[0] > 3.2):
            return 6
        if num_axles == 4 and num_groups > 2 and (2.1<=axles[0] <=3.2) and (axles[1]<=2.1):
            return 7
        if num_axles == 5 and num_groups > 2 and (2.1<=axles[0] <=3.2) and (axles[1]<=2.1):
            return 8
        if (num_axles == 6 and num_groups > 2) or (num_axles > 6 and num_groups == 3):
            return 9

    if is_medium_comb:
        if num_groups == 4 and num_axles > 6:
            return 10
        if (num_groups == 5 or num_groups == 6) and num_axles>6:
            return 11

    if is_long:
        if num_axles > 6 and num_groups > 6:
            return 12
    # if the above logic cannot classify vehicles, return -1 for misc. vehicle type.
    return -1


if __name__ == '__main__':
    axles = [2.85]
    num_groups = calulate_num_groups(axles)
    class_id = vehicle_class_from_axles(axles)
    print(class_id)

