class Solution {
  static String firstUnique(String s) {
    int[] freq = new int[26];
    for (int i = 0; i < s.length(); i++)
      freq[s.charAt(i) - 'a']++;
    for (int i = 0; i < s.length(); i++) {
      if (freq[s.charAt(i) - 'a'] == 1)
        return String.valueOf(s.charAt(i));
    }
    return "";
  }
}
class Test {
  public static void main(String[] args) {
    if (!(Solution.firstUnique("swiss").equals("w") && Solution.firstUnique("aabb").equals(""))) throw new AssertionError("Problem 15");
    System.out.println("passed");
  }
}
