import java.util.*;
class Solution {
  static int[] nextSmaller(int[] a) {
    int[] ans = new int[a.length];
    Deque<Integer> stack = new ArrayDeque<>();
    for (int i = a.length - 1; i >= 0; i--) {
      while (!stack.isEmpty() && stack.peek() >= a[i])
        stack.pop();
      ans[i] = stack.isEmpty() ? -1 : stack.peek();
      stack.push(a[i]);
    }
    return ans;
  }
}
class Test {
  public static void main(String[] args) {
    if (!(java.util.Arrays.equals(Solution.nextSmaller(new int[]{3,2,11,7,6,5,6,1}), new int[]{2,1,7,6,5,1,1,-1}))) throw new AssertionError("Problem 1");
    System.out.println("passed");
  }
}
