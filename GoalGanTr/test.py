# You are given the root of a binary tree and an integer distance.
# A pair of two different leaf nodes of a binary tree is said to be good
# if the length of the shortest path between them is less than or equal to distance.#
# Return the number of good leaf node pairs in the tree.

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def countPairs(self, root: TreeNode, distance: int) -> int:
        count = 0

        def dfs(node):
            nonlocal count
            if not node:
                return []
            if not node.left and not node.right:
                return [1]
            left = dfs(node.left)
            right = dfs(node.right)
            count += sum(l + r <= distance for l in left for r in right)
            return [n + 1 for n in left + right if n + 1 < distance]

        dfs(root)
        return count


## explaination of quick sort:
# 1. partition:
#   1.1. find the pivot
#   1.2. find the index of the pivot
#   1.3. swap the pivot with the last element
#   1.4. find the index of the pivot
#   1.5. swap the pivot with the element at the index
#   1.6. return the index
# 2. recursion:
#   2.1. recursion on the left side
#   2.2. recursion on the right side
#   2.3. return the two parts
# 3. return the two parts



def quickSort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[-1]
    left = []
    right = []
    for i in range(len(arr) - 1):
        if arr[i] < pivot:
            left.append(arr[i])
        else:
            right.append(arr[i])
    return quickSort(left) + [pivot] + quickSort(right)

arr = [5,4,3,1,2,3,4,5]
print(quickSort(arr))
print(arr)