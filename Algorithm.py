import random
import sys
from queue import Queue
import itertools
import collections


class Algorithm:
    """
    选择排序
    https://www.geeksforgeeks.org/selection-sort/
    """
    def selectionsort(self,arr):
        for i in range(len(arr)):
            min_index=i
            for j in range(i+1,len(arr)):
                if arr[j]<arr[min_index]:
                    min_index=j
            if min_index!=i:
                arr[min_index],arr[i]=arr[i],arr[min_index]
        return arr

    """
        冒泡排序
        https://www.geeksforgeeks.org/bubble-sort/
        """
    def bubblesort(self,arr):
        for i in range(len(arr)):
            swap=False
            for j in range(len(arr)-i-1):
                if arr[j]>arr[j+1]:
                    arr[j],arr[j+1]=arr[j+1],arr[j]
                    swap=True
            if not swap:
                break
        return arr

    """
            插入排序
            https://www.geeksforgeeks.org/insertion-sort/
    """
    def insertionsort(self,arr):
        # Traverse through 1 to len(arr)
        for i in range(1, len(arr)):
            key = arr[i]
            # Move elements of arr[0..i-1], that are
            # greater than key, to one position ahead
            # of their current position
            j = i - 1
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    """
            合并排序
            https://www.geeksforgeeks.org/merge-sort/
    """
    # def mergesort(self,arr):

    """
            快速排序
            https://www.geeksforgeeks.org/quick-sort/
    """
    def quicksort(self,arr):
        low=0
        high=len(arr)
        self.quicksortHelper(arr,low,high)
        return arr

    def quicksortHelper(self,arr,low,high):
        if low<high:
            pivot = arr[high-1]#将arr[low:high]中最右边的数作为比较值
            pos = low#pos表示比较值存放的位置，每有一个比比较值小的数，将该值交换到比较值位置，且pos+1，最后交换比较值
            for j in range(low, high-1):
                if arr[j] < pivot:
                    arr[j], arr[pos] = arr[pos], arr[j]
                    pos += 1
            arr[pos], arr[high-1] = arr[high-1], arr[pos]
            #继续遍历比较值位置pos的两边，不包括pos
            self.quicksortHelper(arr,low,pos)
            self.quicksortHelper(arr,pos+1,high)

    """
    二叉树广度度优先BFS 利用queue
    """
    def bfs(self,root):
        if not root:
            return
        q=Queue()
        res=[]
        q.put(root)
        while not q.empty():
            node=q.get()
            res.append(node.val)
            if node.left:
                q.put(node.left)
            if node.right:
                q.put(node.right)
        return res

    """
    二叉树深度优先DFS 中序遍历inorder
    https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/
    """

    def printInorder(self,root):

        if root:
            # First recur on left child
            self.printInorder(root.left)

            # Then print the data of node
            print(root.val, end=" "),

            # Now recur on right child
            self.printInorder(root.right)

    """
    二叉树深度优先 前序遍历preorder
    """

    def printPreorder(self,root):

        if root:
            # Then print the data of node
            print(root.val, end=" "),
            # First recur on left child
            self.printPreorder(root.left)

            # Now recur on right child
            self.printPreorder(root.right)

    """
    二叉树深度优先 后序遍历postorder
    """

    def printPostorder(self,root):

        if root:
            # First recur on left child
            self.printPostorder(root.left)

            # Now recur on right child
            self.printPostorder(root.right)
            # Then print the data of node
            print(root.val, end=" "),

    """
    Algorithm to Solve Sudoku
    """
    def soduke(self,board):
        n=9
        self.board=board
        self.rowlist = [[] for i in range(n)]
        self.columnlist = [[] for j in range(n)]
        self.crosslist = [[] for i in range(n)]
        for i in range(n):
            for j in range(n):
                if board[i][j]!=0:
                    self.rowlist[i].append(board[i][j])
                    self.columnlist[j].append(board[i][j])
                    cross_index=(i//3)*3+j//3
                    self.crosslist[cross_index].append(board[i][j])
        if self.solvesoduke(0,0):
            return board
        else:
            return '000'
    def validnum(self,i,j,num):
        cross_index = (i // 3) * 3 + j // 3
        if num not in self.rowlist[i] and (num not in self.columnlist[j]) and (num not in self.crosslist[cross_index]):
            return True
        return False
    def place_num(self,i,j,num):
        self.board[i][j]=num
        self.rowlist[i].append(num)
        self.columnlist[j].append(num)
        cross_index = (i // 3) * 3 + j // 3
        self.crosslist[cross_index].append(num)
    def delete_num(self,i,j,num):
        self.board[i][j] = 0
        self.rowlist[i].remove(num)
        self.columnlist[j].remove(num)
        cross_index = (i // 3) * 3 + j // 3
        self.crosslist[cross_index].remove(num)
    def solvesoduke(self,i,j):
        if i==9:
            return True
        if self.board[i][j]==0:
            for k in range(1,10):
                if self.validnum(i,j,k):
                    self.place_num(i,j,k)
                    if j==8:
                        new_i=i+1
                        new_j=0
                    else:
                        new_i=i
                        new_j=j+1
                    if self.solvesoduke(new_i,new_j):
                        return True
                    self.delete_num(i,j,k)
        else:
            if j == 8:
                new_i =i+ 1
                new_j = 0
            else:
                new_i = i
                new_j =j+ 1
            return self.solvesoduke(new_i, new_j)

    """
    
    """
    def kighttour(self,num):
        board=[[-1 for j in range(num)] for i in range(num)]
        move_x=[2,2,1,1,-1,-1,-2,-2]
        move_y=[1,-1,2,-2,2,-2,1,-1]
        k_step=1
        board[0][0]=0
        if not self.kighttourBacktracking(num,board,0,0,k_step,move_x,move_y):
            print("Solution does not exist")
        else:
            self.printSolution(num, board)

    def validlocation(self,x,y,num,board):
        if (x>=0 and y>=0 and x<num and y<num and board[x][y]==-1):
            return True
        return False

    def printSolution(self,n, board):
        '''
            A utility function to print Chessboard matrix
        '''
        for i in range(n):
            for j in range(n):
                print(board[i][j], end=' ')
            print()
    def kighttourBacktracking(self,num,board,x_location,y_location,k_step,move_x,move_y):
        if k_step==num**2:
            return True
        for i in range(8):
            new_x = x_location+move_x[i]
            new_y = y_location+move_y[i]
            if self.validlocation(new_x,new_y,num,board):
                board[new_x][new_y]=k_step
                if self.kighttourBacktracking(num,board,new_x,new_y,k_step+1,move_x,move_y):
                    return True
                board[new_x][new_y]=-1
        return False

    """
    
    """
    def mazerat(self,board):
        self.n=len(board)
        self.board=board
        self.grid = [[0 for i in range(self.n)] for j in range(self.n)]
        self.move_x=[1,0]
        self.move_y=[0,1]
        self.grid[0][0]=1
        if self.solvemazerat(0,0):
            return self.grid
        else:
            return 'no valid path'
    def solvemazerat(self,x,y):
        if x==y==self.n-1:
            return True
        for i in range(2):
            new_x=x+self.move_x[i]
            new_y=y+self.move_y[i]
            if (new_x<self.n and new_y<self.n and self.board[new_x][new_y]!=0):
                self.grid[new_x][new_y]=1
                if self.solvemazerat(new_x,new_y):
                    return True
                self.grid[new_x][new_y] = 0
        return False

    def Nqueen(self,n):
        self.n = n
        self.board=[[0 for j in range(n)] for i in range(n)]
        self.q_num=0
        if self.solveNqueen(0):
            for i in range(n):
                for j in range(n):
                    print(self.board[i][j],end=' ')
                print()
        else:
            print('no valid queen')

    def validQueen(self,x,y):
        if x<0 or y<0 or x>=self.n or y>=self.n:
            return False
        for i in self.board[x]:
            if i:
                return False
        for j in range(self.n):
            if self.board[j][y]:
                return False
        for k in range(-self.n,self.n):
            if x+k>=0 and y+k>=0 and x+k<self.n and y+k<self.n:
                if self.board[x+k][y+k]:
                    return False
            if x+k>=0 and y-k>=0 and x+k<self.n and y-k<self.n:
                if self.board[x+k][y-k]:
                    return False
        return True
    def solveNqueen(self,x):
        if self.q_num==self.n:
            return True
        # if y==self.n:
        #     x+=1
        #     y=0
        for i in range(self.n):
            if self.validQueen(x,i):
                self.board[x][i]=1
                self.q_num+=1
                if self.solveNqueen(x+1):
                    return True
                self.board[x][i]=0
                self.q_num-=1
        return False

    # """
    # https://www.geeksforgeeks.org/subset-sum-problem/
    # """
    # def equalpartition(self,targetsum,arr):
    #     n=len(arr)
    #     subset=[]
    #     self.PrintSubsetSum(0, n, arr, targetsum, subset)
    # def PrintSubsetSum(self,point_n,n,arr,targetsum,subset):
    #     if targetsum==0:
    #         print(subset)
    #         return
    #     if point_n==n:
    #         return
    #     self.PrintSubsetSum(point_n+1,n,arr,targetsum,subset)
    #     if arr[point_n]<=targetsum:
    #         subset.append(arr[point_n])
    #         self.PrintSubsetSum(point_n + 1, n, arr, targetsum-arr[point_n], subset)
    #         subset.pop()

    """
    子数列的和等于targetsum，元素可重复
    """
    def CombinationalSum(self,arr,targetsum):
        if targetsum<=0:
            return False
        n=len(arr)
        res_list=[]
        tmp_list=[]
        index=0
        # arr = sorted(list(set(arr)))
        self.solveCombinationalSum(n,arr,targetsum,index,res_list,tmp_list)

        for i in range(len(res_list)):
            for j in range(len(res_list[i])):
                print(res_list[i][j],end=' ')
            print()
        return res_list
    def solveCombinationalSum(self,n,arr,targetsum,index,res_list,tmp_list):
        if targetsum==0:
            res_list.append(list(tmp_list))
            return
        for i in range(index,n):
            if targetsum>=arr[i]:
                tmp_list.append(arr[i])
                self.solveCombinationalSum(n,arr,targetsum-arr[i],i,res_list,tmp_list)
                tmp_list.pop()

    """
    子数列的和等于targetsum，元素不可重复
    """
    def PrintSubsetSum(self,arr,targetsum):
        if targetsum<=0:
            return False
        n=len(arr)
        res_list=[]
        tmp_list=[]
        index=0
        # arr = sorted(list(set(arr)))
        self.solvePrintSubsetSum(n,arr,targetsum,index,res_list,tmp_list)

        for i in range(len(res_list)):
            for j in range(len(res_list[i])):
                print(res_list[i][j],end=' ')
            print()
        return res_list
    def solvePrintSubsetSum(self,n,arr,targetsum,index,res_list,tmp_list):
        if targetsum==0:
            res_list.append(list(tmp_list))
            return
        for i in range(index,n):
            if targetsum>=arr[i]:
                tmp_list.append(arr[i])
                self.solvePrintSubsetSum(n,arr,targetsum-arr[i],i+1,res_list,tmp_list)
                tmp_list.pop()

    """
    生成所有子数列
    """
    def subsetsUtil(self,A, subset, index,res):
        # print(*subset)
        if subset:
            res.append(list(subset))
            return
        for i in range(index, len(A)):
            # include the A[i] in subset.
            subset.append(A[i])

            # move onto the next element.
            self.subsetsUtil(A, subset, i,res)

            # exclude the A[i] from subset and
            # triggers backtracking.
            subset.pop()
        return

    # below function returns the subsets of vector A.
    def subsets(self,A):
        res=[]
        subset = []

        # keeps track of current element in vector A
        index = 0
        self.subsetsUtil(A, subset, index,res)
        for i in range(len(res)):
            for j in range(len(res[i])):
                print(res[i][j],end=' ')
            print()


    """
    dynamic programming
    """
    def MinCostPath(self,map,destination):
        res=[[-1 for j in range(len(map[i]))] for i in range(len(map))]
        res[0][0]=map[0][0]
        result=self.MinCostPathHelper(map,destination,res)
        return result

    def MinCostPathHelper(self,map,destination,res):
        n, m = destination
        if res[n][m]!=-1:
            return res[n][m]
        if n < 0 or m < 0:
            return sys.maxsize
        elif n == 0 and m == 0:
            return res[n][m]
        else:
            res[n][m] = min(self.MinCostPathHelper(map, (n - 1, m),res), self.MinCostPathHelper(map, (n, m - 1),res),
                     self.MinCostPathHelper(map, (n - 1, m - 1),res)) + map[n][m]
        return res[n][m]

    def subsetSumDP(self,arr,sum_a):
        table=[[-1 for i in range(2000)] for j in range(2000)]
        index=len(arr)
        if self.subsetSumDPHelper(arr,sum_a,index,table):
            print('yes')
        else:
            print('false')

    def subsetSumDPHelper(self,arr,sum_a,index,table):
        if sum_a==0:
            return 1
        if index<=0:
            return 0
        if table[index-1][sum_a]!=-1:
            return table[index-1][sum_a]
        if arr[index-1]<=sum_a:
            table[index-1][sum_a]=self.subsetSumDPHelper(arr,sum_a,index-1,table)
            return table[index-1][sum_a] or self.subsetSumDPHelper(arr,sum_a-arr[index-1],index-1,table)
        else:
            table[index - 1][sum_a] = self.subsetSumDPHelper(arr, sum_a, index - 1, table)

    def Fibonacci(self,n):
        if n<2:
            return n
        elif n==2:
            return 1
        else:
            return self.Fibonacci(n-2)+self.Fibonacci(n-1)

    def TowerOfHanoi(self,n, from_rod, to_rod, aux_rod):
        if n == 0:
            return
        self.TowerOfHanoi(n - 1, from_rod, aux_rod, to_rod)
        print("Move disk", n, "from rod", from_rod, "to rod", to_rod)
        self.TowerOfHanoi(n - 1, aux_rod, to_rod, from_rod)

    # def partition(self,arr,n,k):
    #
    # def partitionHelper(self,arr,n,k,res,tmp,index):
    #     if k==0:
    #         res.append(tmp)
    #     if index>n-1:
    #         return
    #     for i in range()
    def test(self,pairs):
        pairs.sort(key=lambda x: x[0])
        n = len(pairs)


    def testbt(self, pairs, index):
        if index<2:
            return index



    """
    leetcode97 
    Update dp[i][j] based on the transition dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or (dp[i][j-1] and s2[j-1] == s3[i+j-1])
    """
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        n = len(s1)
        m = len(s2)
        if n + m != len(s3):
            return False
        dp = [[-1 for i in range(m + 1)] for j in range(n + 1)]
        dp[0][0] = 1
        return self.isInterleaverHelper(s1, s2, s3, n, m,dp)

    def isInterleaverHelper(self, s1, s2, s3, n, m,dp):
        if n < 0 or m < 0:
            return 0
        if dp[n][m]!=-1:
            return dp[n][m]
        dp[n][m] = (self.isInterleaverHelper(s1, s2, s3, n - 1, m,dp) and s1[n - 1] == s3[n + m - 1]) or (
                    self.isInterleaverHelper(s1, s2, s3, n, m - 1,dp) and s2[m - 1] == s3[n + m - 1])
        return dp[n][m]

    """
    646. Maximum Length of Pair Chain
    dp[i]表示pairs[:i+1]所包含的最长递增链。将pairs[i][0]与所有pairs[j][1](0<=j<i)比较，满足pairs[i][0]>pairs[j][1],则取dp[j]+1,否则取1，最后dp[i]等于max(dp[i] or max(dp[j]+1 or 1))(0<=j<i)

    """
    def findLongestChain(self, pairs: list[list[int]]) -> int:
        pairs.sort(key=lambda x: x[0])
        n = len(pairs)
        dp = [1] * n

        for i in range(1, n):
            tmp = []
            for j in range(i):
                if pairs[i][0] > pairs[j][1]:
                    tmp.append(dp[j] + 1)
                else:
                    tmp.append(1)
            dp[i]=max(dp[i],max(tmp))
            # dp[i] = max(dp[i], max(dp[j] + 1 if pairs[i][0] > pairs[j][1] else 1 for j in range(i)))

        return max(dp)

    def isPalindromicString(self,string):
        n=len(string)
        result=-sys.maxsize
        result_s=''
        dp=[[-1 for i in range(n)]for j in range(n)]
        for i in range(n):
            for j in range(n):
                if i<=j:
                    substring=string[i:(j+1)]
                else:
                    substring=string[j:(i+1)]
                sub_re="".join(reversed(substring))
                if sub_re==substring:
                    dp[i][j]=abs(i-j+1)
                else:
                    dp[i][j]=0
                if dp[i][j]>result:
                    result=dp[i][j]
                    result_s=substring
        return result_s

    """
    10. Regular Expression Matching
-
    1, If p.charAt(j) == s.charAt(i) :  dp[i][j] = dp[i-1][j-1];
    2, If p.charAt(j) == '.' : dp[i][j] = dp[i-1][j-1];
    3, If p.charAt(j) == '*': 
        here are two sub conditions:
               1   if p.charAt(j-1) != s.charAt(i) : dp[i][j] = dp[i][j-2]  //in this case, a* only counts as empty
               2   if p.charAt(i-1) == s.charAt(i) or p.charAt(i-1) == '.':
                              dp[i][j] = dp[i-1][j]    //in this case, a* counts as multiple a 
                           or dp[i][j] = dp[i][j-1]   // in this case, a* counts as single a
                           or dp[i][j] = dp[i][j-2]   // in this case, a* counts as empty

    """
    def isMatch(self,text,pattern):
        n = len(text)
        m = len(pattern)
        dp = [[0 for i in range(m + 1)] for j in range(n + 1)]
        dp[0][0] = 1
        for k in range(2, m + 1, 2):
            if pattern[k - 1] == '*':
                dp[0][k] = dp[0][k - 2]
        for i in range(0, n):
            for j in range(0, m):
                if pattern[j] == text[i] or pattern[j] == '.':
                    dp[i + 1][j + 1] = dp[i][j]
                if pattern[j] == '*':
                    if (pattern[j - 1] != text[i]) and (pattern[j - 1] != '.'):
                        dp[i + 1][j + 1] = dp[i + 1][j - 1]
                    else:
                        dp[i + 1][j + 1] = dp[i + 1][j - 1] or dp[i + 1][j] or dp[i][j + 1]
        return dp[n][m]
    def NSum(self,nums,target,n):
        nums.sort()
        res=[]
        tmp=[]
        index=0
        self.NSumbt(nums,target,n,res,tmp,index)
        return res
    def NSumbt(self,nums,target,n,res,tmp,index):
        if n==0 and target==0:
            res.append(list(tmp))
        if index>=len(nums):
            return
        for i in range(index,len(nums)):
            if target>= nums[i]:
                tmp.append(nums[i])
                self.NSumbt(nums,target-nums[i],n-1,res,tmp,i+1)
                tmp.pop()
    """
    403. Frog Jump

    recursion but timeout
    """
    def canCross1(self, stones: list[int]) -> bool:
        if stones[1]!=1:
            return False
        sum=1
        step=1
        return self.canCrossbt(stones,sum,step)


    def canCrossbt(self, stones: list[int],target,step) -> bool:
        if target==stones[-1]:
            return True
        elif target>stones[-1]:
            return False
        if target in stones and step>0:
            return self.canCrossbt(stones,target+step,step) or self.canCrossbt(stones,target+step-1,step-1) or self.canCrossbt(stones,target+step+1,step+1)
        else:
            return False

    """
        403. Frog Jump

        dp but 内存超出
        """
    def canCross2(self, stones: list[int]) -> bool:
        n=stones[-1]
        # dp=[[-1 for j in range(stones[-1]//2+1)]for i in range(len(stones))]
        dp=[[-1 for j in range(n+1)]for i in range(n+1)]

        if stones[0]==0:
            dp[0][0]=True
        # for j in range(stones[-1]//2+1):
        #     dp[0][j]=True
        for i in range(n+1):
            for j in range(n+1):
                if i==j==0 and stones[0]==0:
                    dp[i][j]=True
                elif i==j==1 and stones[1]==1:
                    dp[i][j]=True
                elif i==0:
                    dp[i][j] = True
                elif j==0:
                    dp[i][j]=False
                else:
                    if i in stones:
                        if j>=i:
                            dp[i][j]=dp[i][j-1]
                        else:

                            dp[i][j]=dp[i-j][j] or dp[i-j-1][j] or dp[i-j+1][j] or dp[i][j-1]
        return dp[n][n]

    """
        403. Frog Jump

        dp pass
        Exapme 1:
                    
        index:        0   1   2   3   4   5   6   7 
                    +---+---+---+---+---+---+---+---+
        stone pos:  | 0 | 1 | 3 | 5 | 6 | 8 | 12| 17|
                    +---+---+---+---+---+---+---+---+
        k:          | 1 | 0 | 1 | 1 | 0 | 1 | 3 | 5 |
                    |   | 1 | 2 | 2 | 1 | 2 | 4 | 6 |
                    |   | 2 | 3 | 3 | 2 | 3 | 5 | 7 |
                    |   |   |   |   | 3 | 4 |   |   |
                    |   |   |   |   | 4 |   |   |   |
                    |   |   |   |   |   |   |   |   |
        
        // Sub-problem and state:
        let dp(i) denote a set containing all next jump size at stone i
        
        // Recurrence relation:
        for any j < i,
        dist = stones[i] - stones[j];
        if dist is in dp(j):
            put dist - 1, dist, dist + 1 into dp(i). 
        """
    def canCross3(self, stones: list[int]) -> bool:
        dp = [set() for i in range(len(stones))]
        if stones[0] == 0:
            dp[0].add(1)
        else:
            return False
        for i in range(1, len(stones)):
            for j in range(i):
                k=stones[i]-stones[j]
                if k in dp[j]:
                    dp[i].add(k),dp[i].add(k-1),dp[i].add(k+1)
        if dp[len(stones)-1]:
            return True
        else:
            return False

    """
    18. 4Sum N-Sum
    """
    def NSum(self,arr,target,n):
        arr.sort()
        tmp=[]
        res=[]
        self.NSumHelper(arr,target,n,tmp,res)
        return res
    def NSumHelper(self,arr,target,n,tmp,res):
        if len(arr)<n or n<2 or arr[0]*n>target or arr[-1]*n<target:    # early termination
            return
        if n==2:    # two pointers solve sorted 2-sum problem
            l,r=0,len(arr)-1
            while l<r:
                sum=arr[l]+arr[r]
                if sum==target:
                    res.append(tmp+[arr[l],arr[r]])
                    l+=1
                    while l<r and arr[l]==arr[l-1]:#避免重复
                        l+=1
                elif sum<target:
                    l+=1
                else:
                    r-=1
        else:   # recursively reduce N
            for i in range(len(arr)-n+1):
                if n*arr[i]>target:
                    break
                else:
                    if i == 0 or (i > 0 and arr[i] != arr[i - 1]):#避免重复
                        self.NSumHelper(arr[i+1:],target-arr[i],n-1,tmp+[arr[i]],res)

    """
    5. Longest Palindromic Substring

    Definition : the row and col in the dp table represent the slicing index on the string s (inclusive)
    example s = 'babad' -- > dp[2][3] = s[2:3] = ba
    because i>j,so s[j:i+1]=dp[j][i]
    first s[j]==s[i],then if i-j>=2, dp[j][i]=dp[j+1][i-1], example "cbbc", s[0]=s[3], so if 'bb' is True, 'cbbc' also is True, so dp[0][3]=dp[1][2]
    """

    def longestPalindrome(self, string: str) -> str:
        n = len(string)
        result_s = ''
        dp = [[False for i in range(n)] for j in range(n)]
        for i in range(n):
            dp[i][i] = True
            result_s = string[i]
        for i in range(n):
            for j in range(i - 1, -1, -1): #j从大到小
                tmp_s = ''
                if string[i] == string[j]:
                    if i - j == 1:
                        dp[j][i] = True
                        tmp_s = string[j:(i + 1)]
                    else:
                        dp[j][i] = dp[j + 1][i - 1]
                        if dp[j][i]:
                            tmp_s = string[j:(i + 1)]
                if len(tmp_s) > len(result_s):
                    result_s = tmp_s
        return result_s

    """
    17.Letter Combinations of a Phone Number

    """

    def letterCombinations(self,digits):
        string_list=['abc','def','ghi','jkl','mno','pqrs','tuv','wxyz']
        res=[]
        tmp=''
        if not digits:
            return res
        self.letterCombinationsbt(digits,res,tmp,string_list)
        return res

    def letterCombinationsbt(self,digits,res,tmp,string_list):
        if not digits:
            res.append(tmp)
            return
        for i in string_list[int(digits[0])-2]:
            self.letterCombinationsbt(digits[1:],res,tmp+i,string_list)
    """
    135. Candy
    方法二牛，Go from left to right and while increase, give the the next person +1 candy from previous, if not, leave number of candies as it was. In this way when we make this pass we make sure that condition that person with bigger value gets more candies fulfilled for pairs of adjusent persons where left person is smaller than right. Now, go from right to left and do the same: now we cover pairs of adjacent persons where right is smaller than left. After these two passes all persons are happy.
    """
    def candy(self,ratings):
        if len(ratings) == 1:
            return 1
        dp=[0 for i in range(len(ratings))]
        result=0
        while 0 in dp:
            for i in range(len(ratings)):
                if dp[i]:
                    continue
                if i == 0:
                    if ratings[i] > ratings[i + 1] and dp[i + 1]:
                        dp[i] = dp[i + 1] + 1
                    elif ratings[i] <= ratings[i + 1]:
                        dp[i] = 1
                elif i == len(ratings) - 1:
                    if ratings[i] > ratings[i - 1] and dp[i - 1]:
                        dp[i] = dp[i - 1] + 1
                    elif ratings[i] <= ratings[i - 1]:
                        dp[i] = 1
                else:
                    if ratings[i] > ratings[i + 1] and ratings[i] > ratings[i - 1]:
                        if dp[i + 1] and dp[i - 1]:
                            dp[i] = max(dp[i + 1], dp[i - 1]) + 1
                    elif ratings[i] > ratings[i + 1] and dp[i + 1]:
                        dp[i] = dp[i + 1] + 1
                    elif ratings[i] > ratings[i - 1] and dp[i - 1]:
                        dp[i] = dp[i - 1] + 1
                    elif ratings[i] <= ratings[i + 1] and ratings[i] <= ratings[i - 1]:
                        dp[i] = 1
        for i in dp:
            result+=i
        return result

    def candy1(self, ratings):
        n = len(ratings)
        dp = [1 for i in range(n)]
        for i in range(n - 1):
            if ratings[i + 1] > ratings[i]:
                dp[i + 1] = max(dp[i + 1], dp[i] + 1)
        for j in range(n - 2, -1, -1):
            if ratings[j + 1] < ratings[j]:
                dp[j] = max(dp[j], dp[j + 1] + 1)
        return sum(dp)

    """
    214. Shortest Palindrome

    """

    def shortestPalindrome(self,string):
        n=len(string)
        while n>0:
            substring_reversed=''.join(reversed(string[:n]))
            if substring_reversed==string[:n]:
                break
            n-=1
        left_reversed=''.join(reversed(string[(n):]))
        result = left_reversed+string
        return result

    """
    215. Kth Largest Element in an Array
    快速排序的DAC方法，也可以用选择，冒泡，插入排序等方法
    """

    def findKthLargest(self, nums: list[int], k: int) -> int:
        if not nums:
            return
        pivot=nums[-1]
        right=[]
        left=[]
        mid=[]
        for i in range(len(nums)):
            if nums[i]<pivot:
                right.append(nums[i])
            elif nums[i]==pivot:
                mid.append(nums[i])
            else:
                left.append(nums[i])
        ll=len(left)
        ml=len(mid)
        if ll>=k:
            return self.findKthLargest(left,k)
        elif ll+ml<k:
            return self.findKthLargest(right,k-ll-ml)
        else:
            return mid[0]

    """
    1143. Longest Common Subsequence
    DP方法：dp[i][j]表示t1[:i+1] and t2[:j+1] longest common subsequence
    """

    def longestCommonSubsequence(self,text1,text2):
        n=len(text1)
        m=len(text2)
        dp=[[0 for j in range(m)]for i in range(n)]
        for i in range(m):#first row
            if text1[0]==text2[i]:
                dp[0][i]=1
            elif i and text1[0]!=text2[i]:
                dp[0][i]=dp[0][i-1]
        for j in range(n):#first col
            if text2[0]==text1[j]:
                dp[j][0]=1
            elif j and text2[0]!=text1[j]:
                dp[j][0]=dp[j-1][0]
        for i in range(1,n):
            for j in range(1,m):
                if text1[i]==text2[j]:#字符相同时，采用dp[i-1][j-1]为了避免同一个字符重复计算
                    dp[i][j]=dp[i-1][j-1]+1
                    # dp[i][j]=max(dp[i-1][j],dp[i][j-1])+1
                else:
                    # dp[i][j] = dp[i - 1][j - 1]
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[n-1][m-1]

    """
    53. Maximum Subarray
    DP方法：dp[i][j] means sum(nums[i:j+1]),so when i==j, dp[i][j]=nums[i],and when j>i, dp[i][j]=dp[i][j-1]+nums[j]
    """

    def maxSubArray(self, nums: list[int]) -> int:
        result_arr=[]
        result_sum=-sys.maxsize
        n=len(nums)
        dp = [[0 for j in range(n)]for i in range(n)]
        for i in range(n):
            for j in range(i,n):
                if i==j:
                    dp[i][j]=nums[i]
                else:
                    dp[i][j]=dp[i][j-1]+nums[j]
                if dp[i][j]>result_sum:
                    result_sum=dp[i][j]
                    result_arr=nums[i:j+1]
        return result_arr

    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        dict1 = {}
        for i in strs:
            sort_str=''.join(sorted(i))
            if sort_str in dict1:
                dict1[sort_str].append(i)
            else:
                dict1[sort_str]=[i]
        result=dict1.values()
        return result

    def maxScore(self, cardPoints: list[int], k: int) -> int:
        result=float('-inf')
        if k==len(cardPoints):
            return sum(cardPoints)
        a_card=list(reversed(cardPoints[:k]))+list(reversed(cardPoints[len(cardPoints)-k:]))
        for i in range(len(a_card)-k+1):
            tmp=sum(a_card[i:i+k])
            result=max(tmp,result)
        return result

    """
    2483. Minimum Penalty for a Shop
    DP
    """

    def bestClosingTime(self, customers: str) -> int:
        n=len(customers)
        dp=[[0 for j in range(n)]for i in range(n+1)]
        for i in range(n+1):
            for j in range(n):
                if i>j:#i means closed time, so it is open now
                    if customers[j]=='N':
                        if j:
                            dp[i][j]=dp[i][j-1]+1
                        else:
                            dp[i][j]=1
                    else:
                        if j:
                            dp[i][j]=max(dp[i][j],dp[i][j-1])
                else:#it is closed now
                    if customers[j]=='Y':
                        if j:
                            dp[i][j] = dp[i][j - 1] + 1
                        else:
                            dp[i][j] = 1
                    else:
                        if j:
                            dp[i][j] = max(dp[i][j], dp[i][j - 1])
        min_res=float('inf')
        min_index=0
        for i in range(n+1):
            if dp[i][n-1]<min_res:
                min_res=dp[i][n-1]
                min_index=i
        return min_index

    """
    906. Super Palindromes
    """

    def superpalindromesInRange(self, left: str, right: str) -> int:
        num=0
        left_int=int(left)
        right_int=int(right)
        #平方根
        left_helf=int(left_int**0.5)
        right_helf=int(right_int**0.5)+1
        left_helf_len=len(str(left_helf))
        right_helf_len=len(str(right_helf))
        #遍历的长度，10**0表示长度为1
        l=left_helf_len//2+left_helf_len%2-1
        r = right_helf_len // 2 + right_helf_len % 2
        for i in range(10**l,10**r):
            if len(str(i))*2-1>=left_helf_len:
                #奇数位回文
                tmp_num=int(str(i)+str(i)[:len(str(i))-1][::-1])
                if tmp_num>=left_helf and tmp_num<right_helf:
                    large_tmp=tmp_num**2
                    if str(large_tmp)==str(large_tmp)[::-1] and large_tmp>=left_int and large_tmp<right_int:
                        num+=1
            if len(str(i))*2<=right_helf_len:
                # 偶数位回文
                tmp_num = int(str(i) + str(i)[::-1])
                if tmp_num >= left_helf and tmp_num < right_helf:
                    large_tmp = tmp_num ** 2
                    if str(large_tmp) == str(large_tmp)[::-1] and large_tmp>=left_int and large_tmp<right_int:
                        num += 1
        return num

    """
    54. Spiral Matrix
    """
    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        result=[]
        row=len(matrix)
        column=len(matrix[0])
        up=0
        down=row-1
        left=0
        right=column-1
        while left<=right and up<=down:#上下左右边界判定
            for i in range(left,right+1):
                result.append(matrix[up][i])
            up+=1
            for i in range(up,down+1):
                result.append(matrix[i][right])
            right-=1
            if up<=down:#需要再次判断边界
                for i in range(right,left-1,-1):
                    result.append(matrix[down][i])
            down-=1
            if left <= right:#同上
                for i in range(down,up-1,-1):
                    result.append(matrix[i][left])
            left+=1
        return result

    """
    2366. Minimum Replacements to Sort the Array

    """

    def minimumReplacement(self, nums: list[int]) -> int:
        res=0
        n=len(nums)
        right_num=nums[-1]
        for i in range(n-2,-1,-1): #traverse from right to left
            if nums[i]>right_num:
                times=nums[i]//right_num #means times to divide num
                if not nums[i]%right_num:
                    times-=1 #整除减一次数
                else:
                    right_num=nums[i]//(times+1) #非整除右边最大数为 num//分成的份数
                res+=times
            else:
                right_num = nums[i]
        return res

    """
    994. Rotting Oranges

    """

    def orangesRotting(self, grid: list[list[int]]) -> int:
        # number of rows
        rows = len(grid)
        if rows == 0:  # check if grid is empty
            return -1
        # number of columns
        cols = len(grid[0])
        # keep track of fresh oranges
        fresh_cnt = 0
        # queue with rotten oranges (for BFS)
        rotten = collections.deque()
        # visit each cell in the grid
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    # add the rotten orange coordinates to the queue
                    rotten.append((r, c))
                elif grid[r][c] == 1:
                    # update fresh oranges count
                    fresh_cnt += 1
        # keep track of minutes passed.
        minutes_passed = 0
        # If there are rotten oranges in the queue and there are still fresh oranges in the grid keep looping
        while rotten and fresh_cnt > 0:
            # update the number of minutes passed
            # it is safe to update the minutes by 1, since we visit oranges level by level in BFS traversal.
            minutes_passed += 1
            # process rotten oranges on the current level
            for _ in range(len(rotten)):
                x, y = rotten.popleft()
                # visit all the adjacent cells
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    # calculate the coordinates of the adjacent cell
                    xx, yy = x + dx, y + dy
                    # ignore the cell if it is out of the grid boundary
                    if xx < 0 or xx == rows or yy < 0 or yy == cols:
                        continue
                    # ignore the cell if it is empty '0' or visited before '2'
                    if grid[xx][yy] == 0 or grid[xx][yy] == 2:
                        continue
                    # update the fresh oranges count
                    fresh_cnt -= 1
                    # mark the current fresh orange as rotten
                    grid[xx][yy] = 2
                    # add the current rotten to the queue
                    rotten.append((xx, yy))
        # return the number of minutes taken to make all the fresh oranges to be rotten
        # return -1 if there are fresh oranges left in the grid (there were no adjacent rotten oranges to make them rotten)
        return minutes_passed if fresh_cnt == 0 else -1

    """
    1326. Minimum Number of Taps to Open to Water a Garden

    """

    def minTaps(self, n: int, ranges: list[int]) -> int:
            max_range = [0] * (n + 1)
            # build a list max_range to store the max range it can be watered from each index
            for i, r in enumerate(ranges):
                left, right = max(0, i - r), min(n, i + r)
                max_range[left] = max(max_range[left], right - left)

            # it's a jump game now
            start = end = step = 0
            while end < n:
                step += 1
                start, end = end, max(i + max_range[i] for i in range(start, end + 1))
                if start == end:
                    return -1
            return step
    """
    45. Jump Game II

    """
    def jump(self, nums: list[int]) -> int:
        start = end = step = 0
        n = len(nums) - 1
        while end < n:
            step += 1
            max_num=end
            for i in range(start, end + 1):
                tmp_num=i + nums[i]
                max_num=max(tmp_num,max_num)#max_num means max range in each step
            if max_num==end:
                return False#can not reach end
            else:
                start,end=end,max_num
        return step
    """
    124. Binary Tree Maximum Path Sum

    """
    def maxPathSum(self, root):
        self.max_sum=float('-inf')# place max sum
        self.recursion(root)
        return self.max_sum

    def maxPathSumrecursion(self,root):
        if not root:
            return 0
        mid=root.val
        left_sum=max(0,self.maxPathSumrecursion(root.left))#get left tree sum
        right_sum=max(0,self.maxPathSumrecursion(root.right))#get right tree sum
        tmp_sum=mid+left_sum+right_sum
        self.max_sum=max(tmp_sum,self.max_sum)#比较本地的路线和，不经过父节点
        return mid+max(left_sum,right_sum)#返回最大和到父节点
    def merge(self, nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        if not nums2:
            return nums1
        if not nums1:
            nums1=nums2
            return nums1
        nums1=nums1[:m-n]
        for i in nums2:
            nums1.append(i)
        nums1.sort()
        return nums1

    """
    4. Median of Two Sorted Arrays
    binary search
    https://leetcode.com/problems/median-of-two-sorted-arrays/solutions/2471/very-concise-o-log-min-m-n-iterative-solution-with-detailed-explanation/?envType=daily-question&envId=2023-09-01
    
    """

    def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:
        n=len(nums1)
        m=len(nums2)
        #长度为n的list，可切割位为2n+1，切到偶数l=num[cut//2-1]，r=num[cut//2]，奇数l=r=num[cut//2]
        p_n=2*n+1
        p_m=2*m+1
        #cut1的范围为[0,n+m],使用二分查找，注意cut2=n+m-cut1
        low=0
        high=n+m
        while low<=high:
            mid=(low+high)//2
            cut1=mid
            cut2=n+m-cut1
            if cut1>=p_n:
                high=mid
            elif cut2>=p_m:
                low=mid+1#当high-low=1时，如果low=mid=（low+high）//2=low，low将无法>=high,循环将一直重复
            else:
                if cut1%2:#奇数l=r=num[cut//2]
                    l1=r1=nums1[cut1//2]
                else:#偶数l=num[cut//2-1]，r=num[cut//2]
                    if cut1==0==p_n-1:
                        l1=float('-inf')
                        r1=float('inf')
                    elif cut1==0:
                        l1=float('-inf')
                        r1=nums1[0]
                    elif cut1==p_n-1:
                        l1=nums1[-1]
                        r1=float('inf')
                    else:
                        l1=nums1[cut1//2-1]
                        r1=nums1[cut1//2]
                if cut2%2:
                    l2=r2=nums2[cut2//2]
                else:
                    if cut2==0==p_m-1:
                        l2=float('-inf')
                        r2=float('inf')
                    elif cut2==0:
                        l2=float('-inf')
                        r2=nums2[0]
                    elif cut2==p_m-1:
                        l2=nums2[-1]
                        r2=float('inf')
                    else:
                        l2=nums2[cut2//2-1]
                        r2=nums2[cut2//2]
                #需要满足L1 <= R1 && L1 <= R2 && L2 <= R1 && L2 <= R2
                if l1>r2:
                    high=mid
                elif l2>r1:
                    low=mid+1
                else:
                    return (max(l1,l2)+min(r1,r2))/2
        return -1

    """
    10. Regular Expression Matching
    dp[i][j]means text[:i] can express by pattern[:j]
    """

    def isMatch2(self, text, pattern):
        n=len(text)
        m=len(pattern)
        dp=[[False for i in range(m+1)]for j in range(n+1)]
        #first row dp[0][0] is True, and only second string is * dp[0][j]=dp[0][j-2]
        for j in range(m+1):
            if not j:
                dp[0][j]=True
            elif not j%2:
                if pattern[j-1]=='*':
                    dp[0][j]=dp[0][j-2]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if pattern[j - 1] == '*':
                    #string 和 * 前面的pattern不一致，*表示0，dp[i][j] = dp[i][j - 2]
                    if pattern[j - 2] != text[i - 1] and pattern[j - 2] != '.':
                        dp[i][j] = dp[i][j - 2]
                    #string和 * 前面的pattern一致，*表示0，dp[i][j] = dp[i][j - 2]
                    #*表示1，dp[i][j] = dp[i-1][j - 2]
                    #*表示多个，dp[i][j] = dp[i-1][j]
                    else:
                        dp[i][j] = dp[i - 1][j] or dp[i - 1][j - 2] or dp[i][j - 2]
                elif pattern[j - 1] == '.':
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    if text[i - 1] == pattern[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
        return dp[n][m]

    """
        Question:
            You have an arr(list) including n(int) numbers, return an res(list) including k(int) numbers which is the             smallest sum of arr's subarray.
        Example:
            arr=[1,1,3,3,4,4],n=6,k=8
        Return:
            [1,1,2,3,3,4,4,4]
        Algorithm：
            Create dp[i][j](0<=i<=n,0<=j<=sum(arr)) means how many subarrays of arr[:i] meet sum(subarray)==j
            1.  if i==0 or j==0, dp[i][j]=0
            2.  if j<arr[i-1], dp[i][j]=dp[i-1][j]
                if j==arr[i-1], dp[i][j]=dp[i-1][j]+1
                if j>arr[i-1], dp[i][j]=dp[i-1][j-arr[i-1]]+dp[i-1][j]
        """

    def KSmallestsubarraySum(self,n, k, arr):
        res = []
        arr.sort()
        cols = sum(arr)
        dp = [[-1 for i in range(cols + 1)] for j in range(n + 1)]
        for i in range(n + 1):
            for j in range(cols + 1):
                if i == 0 or j == 0:
                    dp[i][j] = 0
                    continue
                if j < arr[i - 1]:
                    dp[i][j] = dp[i - 1][j]
                elif j == arr[i - 1]:
                    dp[i][j] = dp[i - 1][j] + 1
                else:
                    dp[i][j] = dp[i - 1][j - arr[i - 1]] + dp[i - 1][j]
        for x, y in enumerate(dp[-1]):
            if len(res) >= k:
                break
            if y:
                for i in range(y):
                    if len(res) >= k:
                        break
                    res.append(x)
        return res