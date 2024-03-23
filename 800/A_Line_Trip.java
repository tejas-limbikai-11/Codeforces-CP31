import java.util.Scanner;

/**
 * A_Line_Trip
 */
public class A_Line_Trip {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();

        for(int i=0; i<t; i++) {
            int n = sc.nextInt();
            int x = sc.nextInt();
            int arr[] = new int[n+1];
            arr[0] = 0;
            int maxDiff = 0;

            for(int j=1; j<n+1; j++) {
                arr[j] = sc.nextInt();
            }
            for(int j=0; j<arr.length-1; j++) {
                maxDiff = Math.max(maxDiff, (arr[j+1] - arr[j]));
            }
            int lastDiff = x - arr[arr.length-1];
            if(2 * lastDiff > maxDiff) System.out.println(2 * lastDiff);
            else System.out.println(maxDiff);
        }
    }
}