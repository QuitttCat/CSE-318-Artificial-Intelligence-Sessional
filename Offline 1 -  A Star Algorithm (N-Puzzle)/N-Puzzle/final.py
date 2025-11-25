import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List
from heapq import heappush, heappop
import matplotlib.pyplot as plt
from matplotlib import animation
import os


        
class HeuristicFunction(ABC):
    @abstractmethod
    def calculate_h(self, state) -> float:
        pass

class HammingDistance(HeuristicFunction):
    def calculate_h(self, state) -> float:
        k = state.K
        flat_board = state.board.flatten()
        goal = np.arange(1, k * k)
        goal = np.append(goal, 0)
        mismatches = (flat_board != goal) & (flat_board != 0)
        return np.sum(mismatches)

class ManhattanDistance(HeuristicFunction):
    def calculate_h(self, state) -> float:
        k = state.K
        rows, cols = np.where(state.board != 0)
        values = state.board[rows, cols]
        goal_rows = (values - 1) // k
        goal_cols = (values - 1) % k
        distances = np.abs(rows - goal_rows) + np.abs(cols - goal_cols)
        return np.sum(distances)

class EuclideanDistance(HeuristicFunction):
    def calculate_h(self, state) -> float:
        k = state.K
        rows, cols = np.where(state.board != 0)
        values = state.board[rows, cols]
        goal_rows = (values - 1) // k
        goal_cols = (values - 1) % k
        distances = np.sqrt((rows - goal_rows) ** 2 + (cols - goal_cols) ** 2)
        return np.sum(distances)

class LinearConflict(HeuristicFunction):
    def calculate_h(self, state) -> float:
        k = state.K
        rows, cols = np.where(state.board != 0)
        values = state.board[rows, cols]
        goal_rows = (values - 1) // k
        goal_cols = (values - 1) % k
        manhattan = np.sum(np.abs(rows - goal_rows) + np.abs(cols - goal_cols))
        conflicts = 0
        
        for i in range(k):
            row = state.board[i, :]
            mask = row != 0
            row_values = row[mask]
            goal_rows = (row_values - 1) // k
            valid = goal_rows == i
            row_values = row_values[valid]
            goal_cols = (row_values - 1) % k
            for idx1 in range(len(row_values) - 1):
                for idx2 in range(idx1 + 1, len(row_values)):
                    tk_goal_col, tj_goal_col = goal_cols[idx1], goal_cols[idx2]
                    if tj_goal_col < tk_goal_col:
                        conflicts += 1
        
        for j in range(k):
            col = state.board[:, j]
            mask = col != 0
            col_values = col[mask]
            goal_cols = (col_values - 1) % k
            valid = goal_cols == j
            col_values = col_values[valid]
            goal_rows = (col_values - 1) // k
            for idx1 in range(len(col_values) - 1):
                for idx2 in range(idx1 + 1, len(col_values)):
                    tk_goal_row, tj_goal_row = goal_rows[idx1], goal_rows[idx2]
                    if tj_goal_row < tk_goal_row:
                        conflicts += 1
        return manhattan + 2 * conflicts

class State:
    def __init__(self, K: int, board: np.ndarray, parent = None):
        if K < 3:
            raise ValueError("N must be at least 3 for N-Puzzle.")
        if board.shape != (K, K):
            raise ValueError(f"Board must be {K}x{K}, got {board.shape}.")
        if not np.all(np.sort(board.flatten()) == np.arange(K * K)):
            raise ValueError(f"Board must contain numbers 0 to {K*K-1} exactly once.")
        self.K = K
        self.board = board.copy()
        self.parent = parent
        self.empty_pos = self.empty_cell()

    def empty_cell(self) -> Tuple[int, int]:
        empty = np.where(self.board == 0)
        return (empty[0][0], empty[1][0])
    


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return False
        return np.array_equal(self.board, other.board)
    


    def __hash__(self) -> int:
        return hash(self.board.tobytes())
    

    
    def misplaced_tiles(self) -> int:
        correct_board = np.arange(self.K * self.K).reshape(self.K, self.K)
        return np.sum(self.board != correct_board) - (1 if self.board[self.empty_pos] == 0 else 0)
    


    def __lt__(self, other: 'State') -> bool:
        return self.misplaced_tiles() < other.misplaced_tiles()
    


    def print_state_text(self) -> None:
        for i in range(self.K):
            row = ''
            for j in range(self.K):
                val = self.board[i, j]
                row += f'{val} ' if val != 0 else '0 '
            print(row)
            write_to_output(row)  
        print("")
        write_to_output("") 



def merge_and_count(arr, temp, left, mid, right):
    i = left
    j = mid + 1
    k = left
    inversions = 0
    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp[k] = arr[i]
            i += 1
        else:
            temp[k] = arr[j]
            j += 1
            inversions += (mid - i + 1)
        k += 1
    while i <= mid:
        temp[k] = arr[i]
        i += 1
        k += 1
    while j <= right:
        temp[k] = arr[j]
        j += 1
        k += 1
    arr[left:right + 1] = temp[left:right + 1]
    return inversions

def merge_sort_and_count(arr, temp, left, right):
    inversions = 0
    if left < right:
        mid = left + (right - left) // 2
        inversions += merge_sort_and_count(arr, temp, left, mid)
        inversions += merge_sort_and_count(arr, temp, mid + 1, right)
        inversions += merge_and_count(arr, temp, left, mid, right)
    return inversions



def count_inversions(arr: np.ndarray) :
    arr = np.array(arr[arr != 0], dtype=np.int64)
    temp = np.zeros_like(arr)
    return merge_sort_and_count(arr, temp, 0, len(arr) - 1)



class PuzzleSolver:
    def __init__(self, start_state: State, heuristic: HeuristicFunction):
        self.start_state = start_state
        self.heuristic = heuristic
        self.K = start_state.K
        goal_array = np.append(np.arange(1, self.K * self.K), 0)
        goal_array = goal_array.reshape(self.K, self.K)
        self.goal_state = State(self.K, goal_array)
        self.nodes_explored = 0
        self.nodes_expanded = 0

    def is_solvable(self, state: State) -> bool:
        flat_board = state.board.flatten()
        inversions = count_inversions(flat_board)
        blank_row = state.empty_cell()[0]
        blank_row_from_bottom = self.K - blank_row
        if self.K % 2 == 1:
            return inversions % 2 == 0
        else:
            return (blank_row_from_bottom % 2 == 0 and inversions % 2 == 1) or \
                   (blank_row_from_bottom % 2 == 1 and inversions % 2 == 0)
        

    def reconstruct_path(self, current: State) :
        path = []
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]

    def generate_neighbors(self, state: State) :
        neighbors = []
        row, col = state.empty_cell()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.K and 0 <= new_col < self.K:
                new_board = state.board.copy()
                new_board[row, col], new_board[new_row, new_col] = new_board[new_row, new_col], new_board[row, col]
                neighbor = State(self.K, new_board, parent=state)
                neighbors.append(neighbor)
        return neighbors

    def a_star_search(self) :
        open_list = []
        closed_set = set()
        g_scores = {self.start_state: 0}
        f_scores = {self.start_state: self.heuristic.calculate_h(self.start_state)}
        counter = 0 
        heappush(open_list, (f_scores[self.start_state], self.heuristic.calculate_h(self.start_state), 0, counter, self.start_state))
        
        while open_list:
            _, _, g_score, _, current = heappop(open_list)
            
            
            if current == self.goal_state:
                write_to_output(f"Total Number of Explored Notes: {self.nodes_explored}")
                write_to_output(f"Total Number of Expanded Notes: {self.nodes_expanded}")
                return self.reconstruct_path(current)
            
            closed_set.add(current)
            self.nodes_expanded += 1
            neighbors = self.generate_neighbors(current)
           
            
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                tentative_g_score = g_score + 1
                if neighbor not in f_scores or tentative_g_score + self.heuristic.calculate_h(neighbor) <= f_scores[neighbor]:
                    neighbor.parent = current
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = tentative_g_score + self.heuristic.calculate_h(neighbor)
                    counter += 1
                    heappush(open_list, (f_scores[neighbor], self.heuristic.calculate_h(neighbor), tentative_g_score, counter, neighbor))
                    self.nodes_explored += 1
        
        return None

    def solve(self):
        if not self.is_solvable(self.start_state):
            print("Unsolvable puzzle")
            write_to_output("Unsolvable puzzle")
            return
        
        path = self.a_star_search()
        if path:
            print(f"Minimum number of moves = {len(path) - 1}\n")
            write_to_output(f"Minimum number of moves = {len(path) - 1}\n")
            for i, state in enumerate(path):
                state.print_state_text()

        return path


    def display_solution(self, paths, heuristic_names):
    
        max_frames = max(len(path) for path in paths)

        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        axes = axes.flatten()  
        fig.suptitle("Puzzle Solver - A* Search", fontsize=14)
        
        mats = []
        texts = []
        for idx, (path, heuristic_name, ax) in enumerate(zip(paths, heuristic_names, axes)):
            k = self.K
            mat = ax.matshow(path[0].board, cmap="Blues", vmin=0, vmax=0)
            mats.append(mat)
            ax.set_title(heuristic_name, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            text_grid = [[ax.text(j, i, "", ha="center", va="center", color="black", fontsize=20, fontweight='bold')
                          for j in range(k)] for i in range(k)]
            texts.append(text_grid)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        def update(frame):
            artists = []
            for idx, path in enumerate(paths):
                ax = axes[idx]
                mat = mats[idx]
                text_grid = texts[idx]
                k = self.K
                board = path[min(frame, len(path) - 1)].board
                mat.set_data(board)
                for i in range(k):
                    for j in range(k):
                        val = board[i, j]
                        text_grid[i][j].set_text(str(val) if val != 0 else "")
                artists.append(mat)
                artists.extend(text for row in text_grid for text in row)
            return artists
        
        ani = animation.FuncAnimation(
            fig, update, frames=max_frames, interval=1000, blit=True, repeat=False
        )
        plt.show()


def write_to_output(*args, **kwargs):

    file_path = os.path.join(os.getcwd(), "output.txt")
    with open(file_path, "a", encoding="utf-8") as f:
        print(*args, **kwargs, file=f)
        f.flush()  

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "input.txt")

    file_path = os.path.join(os.getcwd(), "output.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("")  
    
    with open(input_file, 'r', encoding="utf-8") as file:
        k = int(file.readline().strip())
        board = []
        for i in range(k):
            row = list(map(int, file.readline().strip().split()))
            board.append(row)
    
        
    
    board = np.array(board, dtype=int)
    start_state = State(k, board)
    
    paths=[]
    
    heuristics = [
        ("HammingDistance", HammingDistance()),
        ("ManhattanDistance", ManhattanDistance()),
        ("EuclideanDistance", EuclideanDistance()),
        ("LinearConflict", LinearConflict())
    ]
    for heuristic_name, heuristic in heuristics:
        write_to_output(heuristic_name)
        solver = PuzzleSolver(start_state, heuristic)
        path=solver.solve()
        # solver.display_solution(path)
        paths.append(path)
    if all(path is not None for path in paths):
        solver.display_solution(paths, [name for name, _ in heuristics])

    # paths.append(start_state)
    
    for i in range (0,4):
        if(paths[i]!=paths[0]):
            print("Inconsistent")

main()