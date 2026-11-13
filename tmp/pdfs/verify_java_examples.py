import ast
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tmp" / "pdfs" / "build_accenture_guide.py"
WORK = ROOT / "tmp" / "pdfs" / "java_checks"
JAVAC = shutil.which("javac")
JAVA = shutil.which("java")
if not JAVAC or not JAVA:
    raise SystemExit("Java JDK tools not available")

tests = {
    1: "java.util.Arrays.equals(Solution.nextSmaller(new int[]{3,2,11,7,6,5,6,1}), new int[]{2,1,7,6,5,1,1,-1})",
    2: 'Solution.majorityCase("AbCdEfG").equals("ABCDEFG") && Solution.majorityCase("aBCd").equals("abcd")',
    3: "Solution.isPrime(29) && !Solution.isPrime(1) && !Solution.isPrime(49)",
    4: 'Solution.compactLabel("coding").equals("c6g") && Solution.compactLabel("a").equals("a1a")',
    5: 'Solution.countChar("bbacb", \'b\') == 3 && Solution.countChar("AaA", \'a\') == 1',
    6: 'Solution.decodeWays("226") == 3 && Solution.decodeWays("06") == 0 && Solution.decodeWays("10") == 1',
    7: "Solution.leaders(new int[]{16,17,4,3,5,2}).equals(java.util.List.of(17,5,2)) && Solution.leaders(new int[]{5,5}).equals(java.util.List.of(5))",
    8: "Solution.longestAtMostK(new int[]{2,1,3,2,1}, 5) == 2 && Solution.longestAtMostK(new int[]{1,1,1}, 2) == 2",
    14: "java.util.Arrays.equals(Solution.twoSum(new int[]{2,7,11,15}, 9), new int[]{0,1}) && java.util.Arrays.equals(Solution.twoSum(new int[]{3,3}, 6), new int[]{0,1}) && Solution.twoSum(new int[]{1,2}, 10).length == 0",
    15: 'Solution.firstUnique("swiss").equals("w") && Solution.firstUnique("aabb").equals("")',
}

tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
solutions = {}
for node in tree.body:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        continue
    call = node.value
    if not isinstance(call.func, ast.Name):
        continue
    if call.func.id == "problem":
        number = ast.literal_eval(call.args[0])
        if number in tests:
            solutions[number] = ast.literal_eval(call.args[7])
    elif call.func.id == "code" and len(call.args) == 1:
        try:
            value = ast.literal_eval(call.args[0])
        except (ValueError, TypeError):
            continue
        if "static String firstUnique" in value:
            solutions[15] = value

assert set(solutions) == set(tests), (sorted(solutions), sorted(tests))
for number, solution in solutions.items():
    folder = WORK / f"p{number:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    runner = (
        "\nclass Test {\n"
        "  public static void main(String[] args) {\n"
        f"    if (!({tests[number]})) throw new AssertionError(\"Problem {number}\");\n"
        "    System.out.println(\"passed\");\n"
        "  }\n}\n"
    )
    java_file = folder / "Solution.java"
    java_file.write_text(solution + runner, encoding="utf-8")
    subprocess.run([JAVAC, str(java_file)], cwd=folder, check=True, capture_output=True, text=True)
    result = subprocess.run([JAVA, "-cp", str(folder), "Test"], cwd=folder, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "passed"

print(f"Compiled and ran {len(solutions)} Java solution blocks")
