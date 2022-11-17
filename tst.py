import numpy as np
point1 = np.around([1.9,2,3,4]).astype(np.int64)
point2 = np.array([2,6,4,3])
print(point1)
shift_vector = point1-point2
shift_vector = shift_vector/sum(abs(shift_vector))

print(shift_vector)