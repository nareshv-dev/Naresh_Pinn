class Solution {
  static int countChar(String s, char target) {
    int count = 0;
    for (int i = 0; i < s.length(); i++) {
      if (s.charAt(i) == target) count++;
    }
    return count;
  }
}
class Test {
  public static void main(String[] args) {
    if (!(Solution.countChar("bbacb", 'b') == 3 && Solution.countChar("AaA", 'a') == 1)) throw new AssertionError("Problem 5");
    System.out.println("passed");
  }
}
