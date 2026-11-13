import java.util.Locale;
class Solution {
  static String majorityCase(String s) {
    int upper = 0, lower = 0;
    for (char ch : s.toCharArray()) {
      if (ch >= 'A' && ch <= 'Z') upper++;
      else if (ch >= 'a' && ch <= 'z') lower++;
    }
    return upper > lower
        ? s.toUpperCase(Locale.ROOT)
        : s.toLowerCase(Locale.ROOT);
  }
}
class Test {
  public static void main(String[] args) {
    if (!(Solution.majorityCase("AbCdEfG").equals("ABCDEFG") && Solution.majorityCase("aBCd").equals("abcd"))) throw new AssertionError("Problem 2");
    System.out.println("passed");
  }
}
