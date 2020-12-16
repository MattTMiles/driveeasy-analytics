from pydea.algorithms.vehicle_classification import vehicle_class_from_axles

v1_1 = [2.85]
v1_2 = [1.1]
v1_3 = [3.2]
v2_1 = [2.2, 2.2]
v2_2 = [3.1, 2.1]
v2_3 = [2.2, 2.2, 0.4]
v2_4 = [2.1, 2.2, 0.5, 0.5]
v3_1 = [6.3]
v3_2 = [5.5]
v4_1 = [4.2, 1.5]
v4_2 = [1.1, 4.5]
v5_1 = [1.2, 3.2, 1.1]
v5_2 = [3.3, 1.2, 1.1, 0.8]

v1s = [v1_1, v1_2, v1_3]
v2s = [v2_1, v2_2, v2_3, v2_4]
v3s = [v3_1, v3_2]
v4s = [v4_1, v4_2]
v5s = [v5_1, v5_2]


def test_class_1():
    for v in v1s:
        assert vehicle_class_from_axles(v) == 1


def test_class_2():
    for v in v2s:
        print(f'testing vehicle with axle distance: {v}')
        assert vehicle_class_from_axles(v) == 2


def test_class_3():
    for v in v3s:
        print(f'testing vehicle with axle distance: {v}')
        assert vehicle_class_from_axles(v) == 3


def test_class_4():
    for v in v4s:
        print(f'testing vehicle with axle distance: {v}')
        assert vehicle_class_from_axles(v) == 4


def test_class_5():
    for v in v5s:
        print(f'testing vehicle with axle distance: {v}')
        assert vehicle_class_from_axles(v) == 5
