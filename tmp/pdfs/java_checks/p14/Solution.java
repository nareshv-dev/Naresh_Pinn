import java.util.*;
class Solution {
  static int[] twoSum(int[] a, int target) {
    Map<Integer, Integer> seen = new HashMap<>();
    for (int i = 0; i < a.length; i++) {
      long need = (long) target - a[i];
      if (need >= Integer.MIN_VALUE
          && need <= Integer.MAX_VALUE
          && seen.containsKey((int) need))
        return new int[] {seen.get((int) need), i};
      seen.put(a[i], i);
    }
    return new int[0];
  }
}
class Test {
  public static void main(String[] args) {
    if (!(java.util.Arrays.equals(Solution.twoSum(new int[]{2,7,11,15}, 9), new int[]{0,1}) && java.util.Arrays.equals(Solution.twoSum(new int[]{3,3}, 6), new int[]{0,1}) && Solution.twoSum(new int[]{1,2}, 10).length == 0)) throw new AssertionError("Problem 14");
    System.out.println("passed");
  }
}
