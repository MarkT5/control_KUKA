def binary_search(arr, x):
    low, high = 0, len(arr)-1
    mid = 0
    mid_old = 0
    while high >= low:
        mid = (high + low) // 2
        if arr[mid] == x:
            break
        elif arr[mid] > x:
            high = mid
        else:
            low = mid
        if mid_old == high + low:
            break
        mid_old = high + low
    return mid

print(binary_search([1,2,3,6,8,9,11], 10))
