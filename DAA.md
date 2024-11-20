#### Dining Philosophers Problem

```
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class DiningPhilosophers {
    static final int NUM_PHILOSOPHERS = 5;
    static Lock[] forks = new Lock[NUM_PHILOSOPHERS];

    static class Philosopher extends Thread {
        private int id;

        public Philosopher(int id) {
            this.id = id;
        }

        @Override
        public void run() {
            while (true) {
                think();
                eat();
            }
        }

        private void think() {
            System.out.println("Philosopher " + id + " is thinking.");
            try {
                Thread.sleep((long) (Math.random() * 1000));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        private void eat() {
            int leftFork = id;
            int rightFork = (id + 1) % NUM_PHILOSOPHERS;
            if (leftFork < rightFork) {
                forks[leftFork].lock();
                forks[rightFork].lock();
            } else {
                forks[rightFork].lock();
                forks[leftFork].lock();
            }

            try {
                System.out.println("Philosopher " + id + " is eating.");
                try {
                    Thread.sleep((long) (Math.random() * 1000));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            } finally {
                forks[leftFork].unlock();
                forks[rightFork].unlock();
                System.out.println("Philosopher " + id + " finished eating.");
            }
        }
    }

    public static void main(String[] args) {
        for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
            forks[i] = new ReentrantLock();
        }

        Philosopher[] philosophers = new Philosopher[NUM_PHILOSOPHERS];
        for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
            philosophers[i] = new Philosopher(i);
            philosophers[i].start();
        }
    }
}

```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

#### Matrix Multiplication using Multithreading

```
public class MatrixMultiplication implements Runnable{
    private int[][] A;
    private int[][] B;
    private int[][] C;
    private int row;

    public MatrixMultiplication(int[][] A, int[][] B, int[][] C, int row) {
        this.A = A;
        this.B = B;
        this.C = C;
        this.row = row;
    }

    @Override
    public void run() {
        int cols = B[0].length;
        int n = B.length;

        for(int col=0; col<cols; col++) {
            int sum = 0;
            for(int k=0; k<n; k++) {
                sum += A[row][k] * B[k][col];
            }
            C[row][col] = sum;
        }
    }
    public static void main(String[] args) {
        int[][] A = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };
        int[][] B = {
            {9, 8, 7},
            {6, 5, 4},
            {3, 2, 1}
        };
        int rows = A.length;
        int cols = B[0].length;
        int[][] C = new int[rows][cols];

        Thread[] threads = new Thread[rows];
        for(int i=0; i<rows; i++) {
            threads[i] = new Thread(new MatrixMultiplication(A, B, C, i));
            threads[i].start();
        }

        for(int i=0; i<rows; i++) {
            try {
                threads[i].join();
            } catch(InterruptedException e) {
                e.printStackTrace();
            }
        }

        System.out.println("Result of matrix multiplication:");
        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                System.out.print(C[i][j] + " ");
            }
            System.out.println();
        }
    }
}
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

#### TSP Branch and Bound

```
public class TSPBranchAndBound {
    public static void main(String[] args) {
        int[][] graph = {
            {0, 10, 15, 20},
            {10, 0, 35, 25},
            {15, 35, 0, 30},
            {20, 25, 30, 0}
        };
        System.out.println("Min cost is: " + tsp(graph));
    }

    public static int tsp(int[][] graph) {
        int n = graph.length;
        boolean visited[] = new boolean[n];
        visited[0] = true;

        int[] minCost = {Integer.MAX_VALUE};
        branchAndBound(0, 1, 0, minCost, graph, visited);
        return minCost[0];
    }

    public static void branchAndBound(int currPos, int count, int cost, int[] minCost, int[][] graph, boolean[] visited) {
        int n = graph.length;

        if(count == n && graph[currPos][0] > 0) {
            minCost[0] = Math.min(minCost[0], cost + graph[currPos][0]);
            return;
        }

        for(int next=0; next<n; next++) {
            if(!visited[next] && graph[currPos][next] > 0) {
                int newCost = cost + graph[currPos][next];

                if(newCost >= minCost[0]) continue;

                visited[next] = true;
                branchAndBound(next, count + 1, newCost, minCost, graph, visited);
                visited[next] = false;
            }
        }
    }
}
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

#### N Queens

```
import java.util.*;

public class NQueens {
    public static void main(String[] args) {
        int n = 8;
        char[][] board = new char[n][n];
        for(char[] ch: board) Arrays.fill(ch, '.');

        if(nQueens(board, 0)) {
            System.out.println("Solution is possible");
            printBoard(board);
        }
        else System.out.println("Solution is not possible");
    }

    public static boolean nQueens(char[][] board, int row) {
        if(row == board.length) {
            return true;
        }

        for(int j=0; j<board.length; j++) {
            if(isSafe(board, row, j)) {
                board[row][j] = 'Q';
                if(nQueens(board, row + 1)) {
                    return true;
                }
                board[row][j] = '.';
            }
        }
        return false;
    }

    public static boolean isSafe(char[][] board, int row, int col) {
        //vertical
        for(int i=row-1; i>=0; i--) {
            if(board[i][col] == 'Q') return false;
        }

        //diagonal left
        for(int i=row-1, j=col-1; i >= 0 && j >= 0; i--, j--) {
            if(board[i][j] == 'Q') return false;
        }

        //diagonal right
        for(int i=row-1, j=col+1; i >= 0 && j < board.length; i--, j++) {
            if(board[i][j] == 'Q') return false;
        }

        return true;
    }

    public static void printBoard(char[][] board) {
        System.out.println("-----Chess Board-----");
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board.length; j++) {
                System.out.print(board[i][j] + " ");
            }
            System.out.println();
        }
    }
}
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

#### MergeSort

```
public class MergeSort {
    public static void main(String[] args) {
        int arr[] = {4, 1, 3, 9, 7};
        mergeSort(arr,0, arr.length-1);

        for(int num: arr) System.out.print(num + " ");
    }

    public static void mergeSort(int arr[], int l, int r) {
        if(l >= r) return;

        int mid = l + (r - l) / 2;
        mergeSort(arr, l, mid);
        mergeSort(arr, mid + 1, r);
        merge(arr, l, mid, r);
    }

    public static void merge(int arr[], int l, int mid, int r) {
        int n1 = mid - l + 1;
        int n2 = r - mid;

        int L[] = new int[n1];
        int R[] = new int[n2];

        int k = l;

        for(int i=0; i<n1; i++) {
            L[i] = arr[k];
            k++;
        }

        for(int i=0; i<n2; i++) {
            R[i] = arr[k];
            k++;
        }

        int i = 0;
        int j = 0;
        k = l;

        while(i < n1 && j < n2) {
            if(L[i] <= R[j]) {
                arr[k] = L[i];
                i++;
            }
            else {
                arr[k] = R[j];
                j++;
            }
            k++;
        }

        while(i < n1) {
            arr[k] = L[i];
            i++;
            k++;
        }

        while(j < n2) {
            arr[k] = R[j];
            j++;
            k++;
        }
    }
}
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

#### QuickSort  

```
public class QuickSort {
    public static void main(String[] args) {
        int arr[] = {2, 1, 6, 10, 4, 1, 3, 9, 7};
        quickSort(arr, 0, arr.length-1);

        for(int num: arr) System.out.print(num + " ");
    }

    public static void quickSort(int[] arr, int low, int high) {
        if(low >= high) return;

        int pivotIdx = partition(arr, low, high);
        quickSort(arr, low, pivotIdx - 1);
        quickSort(arr, pivotIdx + 1, high);
    }

    public static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int pivotIdx = low;

        for(int i=low; i<high; i++) {
            if(arr[i] <= pivot) {
                swap(arr, i, pivotIdx);
                pivotIdx++;
            }
        }

        swap(arr, pivotIdx, high);
        return pivotIdx;
    }

    public static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
#### Knapsack

```
public class Knapsack {
    public static void main(String[] args) {
        int W = 5;
        int val[] = {10, 40, 30, 50};
        int wt[] = {5, 4, 6, 3};
        System.out.println(knapSack(W, val, wt));
    }

    static int knapSack(int W, int val[], int wt[]) {
        int n = wt.length;
        int dp[][] = new int[n+1][W+1];

        for(int i=0; i<n+1; i++) {
            for(int j=0; j<W+1; j++) {
                if(i == 0 || j == 0) dp[i][j] = 0;
            }
        }

        for(int i=1; i<n+1; i++) {
            for(int j=1; j<W+1; j++) {
                if(wt[i-1] <= j) {
                    int take = val[i-1] + dp[i-1][j - wt[i-1]];
                    int notTake = dp[i-1][j];
                    dp[i][j] = Math.max(take, notTake);
                }
                else dp[i][j] = dp[i-1][j];
            }
        }
        return dp[n][W];
    }
}
```
________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

#### TSP Genetic Algorithm

```
import java.util.*;

class City {
    int x, y;

    public City(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public double distanceTo(City other) {
        return Math.hypot(this.x - other.x, this.y - other.y);
    }
}

class Tour {
    private List<City> cities;
    private double distance = 0;

    public Tour(List<City> cities) {
        this.cities = new ArrayList<>(cities);
        Collections.shuffle(this.cities);
    }

    public double getDistance() {
        if (distance == 0) {
            for (int i = 0; i < cities.size(); i++) {
                City from = cities.get(i);
                City to = cities.get((i + 1) % cities.size());
                distance += from.distanceTo(to);
            }
        }
        return distance;
    }

    public List<City> getCities() {
        return cities;
    }

    public void mutate() {
        Collections.swap(cities, new Random().nextInt(cities.size()), new Random().nextInt(cities.size()));
    }

    public static Tour crossover(Tour parent1, Tour parent2) {
        List<City> child = new ArrayList<>(Collections.nCopies(parent1.cities.size(), null));
        int start = new Random().nextInt(parent1.cities.size());
        int end = new Random().nextInt(parent1.cities.size());
        if (start > end) { int temp = start; start = end; end = temp; }

        for (int i = start; i <= end; i++) {
            child.set(i, parent1.cities.get(i));
        }

        for (City city : parent2.cities) {
            if (!child.contains(city)) {
                child.set(child.indexOf(null), city);
            }
        }
        return new Tour(child);
    }
}

class GeneticAlgorithm {
    private List<Tour> population;
    private double mutationRate;

    public GeneticAlgorithm(List<City> cities, int populationSize, double mutationRate) {
        this.mutationRate = mutationRate;
        population = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            population.add(new Tour(cities));
        }
    }

    public Tour evolve(int generations) {
        for (int i = 0; i < generations; i++) {
            List<Tour> newPopulation = new ArrayList<>();
            for (int j = 0; j < population.size(); j++) {
                Tour parent1 = selectParent(), parent2 = selectParent();
                Tour child = Tour.crossover(parent1, parent2);
                if (Math.random() < mutationRate) child.mutate();
                newPopulation.add(child);
            }
            population = newPopulation;
        }
        return population.stream().min(Comparator.comparing(Tour::getDistance)).orElse(null);
    }

    private Tour selectParent() {
        return Collections.min(
                new Random().ints(5, 0, population.size())
                        .mapToObj(population::get)
                        .toList(),
                Comparator.comparing(Tour::getDistance)
        );
    }
}

public class Main {
    public static void main(String[] args) {
        List<City> cities = Arrays.asList(
                new City(60, 200), new City(180, 200), new City(80, 180),
                new City(140, 180), new City(20, 160), new City(100, 160),
                new City(200, 160), new City(140, 140), new City(40, 120), new City(100, 120)
        );

        GeneticAlgorithm ga = new GeneticAlgorithm(cities, 100, 0.01);
        Tour bestTour = ga.evolve(500);

        System.out.println("Best tour found:");
        bestTour.getCities().forEach(city -> System.out.println("(" + city.x + ", " + city.y + ")"));
        System.out.println("Total distance: " + bestTour.getDistance());
    }
}
```
