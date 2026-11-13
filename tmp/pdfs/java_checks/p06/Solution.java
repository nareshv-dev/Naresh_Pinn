class Solution {
  static long decodeWays(String s) {
    if (s == null || s.isEmpty() || s.charAt(0) == '0')
      return 0;
    long prev2 = 1, prev1 = 1;
    for (int i = 1; i < s.length(); i++) {
      long current = 0;
      if (s.charAt(i) != '0') current += prev1;
      int pair = (s.charAt(i - 1) - '0') * 10
          + (s.charAt(i) - '0');
      if (pair >= 10 && pair <= 26) current += prev2;
      prev2 = prev1;
      prev1 = current;
    }
    return prev1;
  }
}
class Test {
  public static void main(String[] args) {
    if (!(Solution.decodeWays("226") == 3 && Solution.decodeWays("06") == 0 && Solution.decodeWays("10") == 1)) throw new AssertionError("Problem 6");
    System.out.println("passed");
  }
}
