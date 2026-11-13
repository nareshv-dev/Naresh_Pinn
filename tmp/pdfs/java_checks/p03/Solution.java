class Solution {
  static boolean isPrime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int d = 3; d <= n / d; d += 2) {
      if (n % d == 0) return false;
    }
    return true;
  }
}
class Test {
  public static void main(String[] args) {
    if (!(Solution.isPrime(29) && !Solution.isPrime(1) && !Solution.isPrime(49))) throw new AssertionError("Problem 3");
    System.out.println("passed");
  }
}
