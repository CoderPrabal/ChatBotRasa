data_list = int(input("Enter the numbers you want to enter"))
global_input_list = []
for i in range(0, data_list):
    input_value = int(input("Enter the number"))
    global_input_list.append(input_value)


def binary_search(list_value):
    value = int(input("Enter the number you want to search"))
    print("The value of the list is",sorted(list_value))
    list_value=sorted(list_value)
    low = 0
    print("The value of low is ",low)
    high = len(list_value) - 1
    print("The value of high is",high)
    while low <= high:
        mid = int((low + high) / 2)
        print("The value of the middle", mid)
        if value == list_value[mid]:
            print("The number is found at pos", mid+1)
            break
        elif value > list_value[mid]:
            low = mid + 1
            print("The value of low is", low)
        elif value < list_value[mid]:
            high = mid - 1
            print("The value of high is", high)


binary_search(global_input_list)
