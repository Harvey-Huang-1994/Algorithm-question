# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

from Leetcode import *
from Algorithm import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # data = [[2,4,3],[5,6,4]]
    # listnode_list=[]
    # for i in range(len(data)):
    #     tail = head = ListNode(data[i][0])
    #     for x in data[i][1:]:
    #         tail.next = ListNode(x)
    #         tail = tail.next
    #     listnode_list.append(head)
    # root=TreeNode(1)
    # left=TreeNode(2)
    # right=TreeNode(3)
    # root.left=left
    # root.right=right
    a_result = Algorithm().KSmallestsubarraySum(6,8,[1,1,3,3,4,4])
    print(a_result)
