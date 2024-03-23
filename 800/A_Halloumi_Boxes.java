import java.util.Scanner;

/**
 * A_Halloumi_Boxes
 */
public class A_Halloumi_Boxes {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        
        for(int i=0; i<t; i++) {
            int n = sc.nextInt();
            int k = sc.nextInt();
            int arr[] = new int[n];
            
            for(int j=0; j<n; j++) {
                arr[j] = sc.nextInt();
            }

            if(k == 1) {
                if(isSorted(arr)) {
                    System.out.println("YES");
                }
                else {
                    System.out.println("NO");
                }
            }
            else System.out.println("YES");
        }
    }

    public static boolean isSorted(int arr[]) {
        for(int i=0; i<arr.length-1; i++) {
            if(arr[i] > arr[i+1]) {
                return false;
            }
        }
        return true;
    }
}