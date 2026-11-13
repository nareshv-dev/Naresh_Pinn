class Solution {
  static int longestAtMostK(int[] a, long k) {
    int left = 0, best = 0;
    long sum = 0;
    for (int right = 0; right < a.length; right++) {
      sum += a[right];
      while (sum > k && left <= right) {
        sum -= a[left++];
      }
      best = Math.max(best, right - left + 1);
    }
    return best;
  }
}
class Test {
  public static void main(String[] args) {
    if (!(Solution.longestAtMostK(new int[]{2,1,3,2,1}, 5) == 2 && Solution.longestAtMostK(new int[]{1,1,1}, 2) == 2)) throw new AssertionError("Problem 8");
    System.out.println("passed");
  }
}
