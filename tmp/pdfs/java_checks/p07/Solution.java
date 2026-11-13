import java.util.*;
class Solution {
  static List<Integer> leaders(int[] a) {
    List<Integer> found = new ArrayList<>();
    long maximum = Long.MIN_VALUE;
    for (int i = a.length - 1; i >= 0; i--) {
      if (a[i] > maximum) {
        found.add(a[i]);
        maximum = a[i];
      }
    }
    Collections.reverse(found);
    return found;
  }
}
class Test {
  public static void main(String[] args) {
    if (!(Solution.leaders(new int[]{16,17,4,3,5,2}).equals(java.util.List.of(17,5,2)) && Solution.leaders(new int[]{5,5}).equals(java.util.List.of(5)))) throw new AssertionError("Problem 7");
    System.out.println("passed");
  }
}
