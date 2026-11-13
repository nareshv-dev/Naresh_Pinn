class Solution {
  static String compactLabel(String s) {
    if (s == null || s.isEmpty())
      throw new IllegalArgumentException("nonempty string required");
    return "" + s.charAt(0) + s.length()
        + s.charAt(s.length() - 1);
  }
}
class Test {
  public static void main(String[] args) {
    if (!(Solution.compactLabel("coding").equals("c6g") && Solution.compactLabel("a").equals("a1a"))) throw new AssertionError("Problem 4");
    System.out.println("passed");
  }
}
