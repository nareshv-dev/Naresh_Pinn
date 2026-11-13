from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted,
    KeepTogether, Table, TableStyle, HRFlowable
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf" / "Accenture_Campus_Coding_Questions_Java_Edition.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#14243D")
BLUE = colors.HexColor("#2467A6")
TEAL = colors.HexColor("#0B8F87")
PALE = colors.HexColor("#EAF2F8")
LIGHT = colors.HexColor("#F4F7FA")
MUTED = colors.HexColor("#526174")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleX", fontName="Helvetica-Bold", fontSize=24, leading=29, textColor=NAVY, spaceAfter=16))
styles.add(ParagraphStyle(name="SubX", fontName="Helvetica", fontSize=11, leading=16, textColor=MUTED, spaceAfter=12))
styles.add(ParagraphStyle(name="H1X", fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=NAVY, spaceBefore=4, spaceAfter=10))
styles.add(ParagraphStyle(name="H2X", fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=BLUE, spaceBefore=8, spaceAfter=5))
styles.add(ParagraphStyle(name="BodyX", fontName="Helvetica", fontSize=9.3, leading=14, textColor=NAVY, spaceAfter=7))
styles.add(ParagraphStyle(name="SmallX", fontName="Helvetica", fontSize=8, leading=11, textColor=MUTED, spaceAfter=6))
styles.add(ParagraphStyle(name="TagX", fontName="Helvetica-Bold", fontSize=8.1, leading=11, textColor=TEAL, spaceAfter=5))
styles.add(ParagraphStyle(name="CodeX", fontName="Courier", fontSize=8, leading=11, textColor=NAVY, leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=9, backColor=LIGHT, borderPadding=8))
styles.add(ParagraphStyle(name="TCell", fontName="Helvetica", fontSize=8.5, leading=11, textColor=NAVY))
styles.add(ParagraphStyle(name="THead", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=colors.white))

story = []


def P(text, style="BodyX"):
    return Paragraph(text, styles[style])


def add(text, style="BodyX"):
    story.append(P(text, style))


def code(text):
    story.append(Preformatted(text.strip("\n"), styles["CodeX"], maxLineLength=92))


def bullet(text):
    add("&#8226; " + text)


def section(title):
    story.append(PageBreak())
    add(title, "H1X")
    story.append(HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceAfter=10))


def problem(number, title, tag, source, prompt, example, idea, solution, complexity, checks, language="Java 17"):
    story.append(PageBreak())
    add(f"{number:02d}  {title}", "H1X")
    add(tag.upper(), "TagX")
    add(f"Source: {escape(source)}", "SmallX")
    add("Problem", "H2X")
    add(prompt)
    add("Example", "H2X")
    code(example)
    add("How to think", "H2X")
    add(idea)
    add(f"Answer code - {language}", "H2X")
    code(solution)
    add("Complexity", "H2X")
    add(complexity)
    add("Check yourself", "H2X")
    add(checks, "SmallX")


def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(colors.HexColor("#DDE5EC"))
    canvas.line(17*mm, h-14*mm, w-17*mm, h-14*mm)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(BLUE)
    canvas.drawString(17*mm, h-11*mm, "ACCENTURE CAMPUS CODING | STUDY GUIDE")
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(17*mm, 12*mm, "Candidate reports + original worked practice | 23 Sep 2026")
    canvas.drawRightString(w-17*mm, 12*mm, str(doc.page))
    canvas.restoreState()


add("Accenture Campus Coding Questions - Java Edition", "TitleX")
add("A student-friendly worked guide: reported previous-year questions, code, SQL, frontend, timed practice, and source notes.", "SubX")
story.append(Spacer(1, 15*mm))

cover = [
    [P("Inside", "THead"), P("What you will learn", "THead")],
    [P("Part A", "TCell"), P("Questions explicitly described by candidates in 2023-2025 / 2026-batch campus reports", "TCell")],
    [P("Part B", "TCell"), P("Original SQL and frontend exercises matching the repeatedly reported three-part format", "TCell")],
    [P("Part C", "TCell"), P("A 60-minute mock, answer checklist, revision schedule, and credible source links", "TCell")],
]
t = Table(cover, colWidths=[29*mm, 145*mm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), NAVY),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#D9E3ED")),
    ("INNERGRID", (0,0), (-1,-1), 0.3, colors.HexColor("#D9E3ED")),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
    ("RIGHTPADDING", (0,0), (-1,-1), 8),
    ("TOPPADDING", (0,0), (-1,-1), 8),
    ("BOTTOMPADDING", (0,0), (-1,-1), 8),
]))
story.append(t)
story.append(Spacer(1, 15*mm))
add("Read this first", "H2X")
add("Accenture does not publish a complete official bank of past test questions. Candidate reports may omit constraints or recall a question approximately. The reported questions here are paraphrased and solved for a clearly stated version. The SQL/frontend practice tasks are original exercises, not claims of an exact leaked paper.")
add("Answers use Java 17 for algorithm problems, standard SQL for database tasks, and plain HTML/CSS/JavaScript for web tasks. Java snippets show the solution class and method; use the exact method signature or input/output wrapper required by your assessment editor.")

section("How hard is it to clear?")
add("For a student comfortable with basic programming, many reported tasks are easy to medium. Clearing all rounds still takes preparation: the current campus process can test cognitive speed, technical MCQs, DSA, SQL, frontend work, communication, and interview readiness. Difficulty and cutoffs change by role and campus. No practice PDF can guarantee selection.")
add("Reported patterns", "H2X")
rows = [
    [P("Period", "THead"), P("Reported coding format", "THead"), P("Use this guide for", "THead")],
    [P("2023-2024 drives", "TCell"), P("Often 2 algorithm questions in about 45 minutes", "TCell"), P("Problems 1-6 and 60-minute mock DSA", "TCell")],
    [P("2025-2026-batch drives", "TCell"), P("Often 1 DSA + 1 SQL + 1 web task in about 60 minutes", "TCell"), P("Problems 7-14 and the three-part mock", "TCell")],
    [P("Other variants", "TCell"), P("Coding and aptitude order or number of tasks may differ", "TCell"), P("Use your campus invitation as final authority", "TCell")],
]
t=Table(rows,colWidths=[33*mm,72*mm,69*mm])
t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),("VALIGN",(0,0),(-1,-1),"TOP"),("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#D9E3ED")),("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#D9E3ED")),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
story.append(t)
add("A reliable attempt strategy", "H2X")
bullet("Read the full task, examples, constraints, function signature, and expected return type.")
bullet("For the three-part format, budget roughly 25 minutes DSA, 15 minutes SQL, and 20 minutes frontend; adjust to your strengths.")
bullet("Test edge cases: empty/single item, duplicates, zeros, mixed case, missing SQL matches, and repeated button clicks.")
bullet("Submit working code; visible examples alone do not prove hidden cases pass.")

section("Question map and evidence labels")
add("Reported = a candidate explicitly described the question or its central requirement. Reconstructed = the report named the topic, but our exact constraints and examples are supplied for learning. Practice = original exercise designed for the reported format.")
items = [
    ("01", "Next smaller element to the right", "Reported, 2023"),
    ("02", "Majority letter case", "Reported, 2023"),
    ("03", "Prime check", "Reported, 2024"),
    ("04", "First character + length + last", "Reported, 2024"),
    ("05", "Count character occurrences", "Reported, 2025"),
    ("06", "Decode Ways", "Reported topic, 2025; reconstructed details"),
    ("07", "Leader elements", "Reported topic, 2025; reconstructed details"),
    ("08", "Dynamic sliding window", "Reported topic, 2025; practice version"),
    ("09", "Two-table SQL join", "Reported topic, 2025; practice schema"),
    ("10", "Three-table SQL join", "Reported topic, 2025; practice schema"),
    ("11", "SQL grouping and filtering", "Practice"),
    ("12", "DOM button interaction", "Reported task type, 2025; practice version"),
    ("13", "Form validation and layout", "Practice"),
    ("14", "Two Sum with a hash map", "Practice for reported hash-map tasks"),
]
for n, title, label in items:
    add(f"<b>{n}</b> &nbsp; {escape(title)} &nbsp; <font color='#0B8F87'>[{escape(label)}]</font>")
add("Use Problems 1-8 and 14 to build coding fluency. Use Problems 9-13 for the newer multi-domain assessment.", "SmallX")

problem(1,"Next smaller element to the right","Reported question - 2023","GeeksforGeeks candidate report [S1]",
"For each array value, return the first value to its right that is strictly smaller. Use -1 when no such value exists. This is the standard next-smaller-element interpretation of the reported prompt.",
"Input:  [3, 2, 11, 7, 6, 5, 6, 1]\nOutput: [2, 1, 7, 6, 5, 1, 1, -1]",
"Scan from right to left. A stack keeps possible answers. Remove values greater than or equal to the current value; the top that remains is the nearest smaller candidate.",
"import java.util.*;\nclass Solution {\n  static int[] nextSmaller(int[] a) {\n    int[] ans = new int[a.length];\n    Deque<Integer> stack = new ArrayDeque<>();\n    for (int i = a.length - 1; i >= 0; i--) {\n      while (!stack.isEmpty() && stack.peek() >= a[i])\n        stack.pop();\n      ans[i] = stack.isEmpty() ? -1 : stack.peek();\n      stack.push(a[i]);\n    }\n    return ans;\n  }\n}",
"O(n) time and O(n) extra space. Each value enters and leaves the stack at most once.",
"Try [4, 4, 3] - answer [3, 3, -1]. Try [] - answer [].")

problem(2,"Majority letter case","Reported question - 2023","GeeksforGeeks candidate report [S1]",
"Count uppercase and lowercase English letters in a string. Convert the entire string to uppercase if uppercase letters are more numerous; otherwise convert to lowercase. The tie rule follows the candidate's published solution.",
"Input:  AbCdEfG\nOutput: ABCDEFG\n\nInput:  aBCd\nOutput: abcd",
"Count first, then use the language's built-in conversion. Read the tie rule carefully; a different drive may specify another rule.",
"import java.util.Locale;\nclass Solution {\n  static String majorityCase(String s) {\n    int upper = 0, lower = 0;\n    for (char ch : s.toCharArray()) {\n      if (ch >= 'A' && ch <= 'Z') upper++;\n      else if (ch >= 'a' && ch <= 'z') lower++;\n    }\n    return upper > lower\n        ? s.toUpperCase(Locale.ROOT)\n        : s.toLowerCase(Locale.ROOT);\n  }\n}",
"O(n) time and O(n) space for the returned string.",
"For 'A1b', counts tie, so the result is 'a1b' under this stated rule.")

problem(3,"Check whether a number is prime","Reported question - 2024","GeeksforGeeks 2025 on-campus experience, assessment in 2024 [S2]",
"Return True if integer n is prime. A prime is greater than 1 and has no positive divisors except 1 and itself.",
"Input:  29\nOutput: True\n\nInput:  1\nOutput: False",
"A composite number has a divisor no larger than its square root. Test possible divisors while d*d <= n.",
"class Solution {\n  static boolean isPrime(int n) {\n    if (n < 2) return false;\n    if (n == 2) return true;\n    if (n % 2 == 0) return false;\n    for (int d = 3; d <= n / d; d += 2) {\n      if (n % d == 0) return false;\n    }\n    return true;\n  }\n}",
"O(sqrt(n)) time and O(1) extra space.",
"Check 2, 3, 4, 9, 97, 0, and negative numbers.")

problem(4,"First character + length + last","Reported question - 2024","GeeksforGeeks 2025 on-campus experience, assessment in 2024 [S2]",
"Given a nonempty string, create a new string containing its first character, its length in decimal, and its last character. We state the nonempty constraint because the report did not publish edge-case rules.",
"Input:  coding\nOutput: c6g\n\nInput:  a\nOutput: a1a",
"Index the first and last characters and convert the integer length to text. Do not confuse length with the number of characters between the endpoints.",
"class Solution {\n  static String compactLabel(String s) {\n    if (s == null || s.isEmpty())\n      throw new IllegalArgumentException(\"nonempty string required\");\n    return \"\" + s.charAt(0) + s.length()\n        + s.charAt(s.length() - 1);\n  }\n}",
"O(1) indexing plus O(log n) characters to represent the length; O(log n) output overhead.",
"Try a two-character string and a 10-character string.")

problem(5,"Count character occurrences","Reported question - 2025","Naukri Code360 December 2025 campus candidate report [S3]",
"Given a string s and a character target, return how many times target appears in s. This version is case sensitive, matching the example's direct equality behavior.",
"Input:  s='bbacb', target='b'\nOutput: 3",
"Walk through the string once. Increment a counter only on an exact match.",
"class Solution {\n  static int countChar(String s, char target) {\n    int count = 0;\n    for (int i = 0; i < s.length(); i++) {\n      if (s.charAt(i) == target) count++;\n    }\n    return count;\n  }\n}",
"O(n) time and O(1) extra space.",
"Try an empty string and a target that never appears.")

problem(6,"Decode Ways","Reported topic - 2025; reconstructed prompt","Naukri Code360 October 2025 campus candidate report [S4]",
"Digits '1' through '26' map to letters A through Z. Count how many ways a nonempty digit string can be decoded. A standalone '0' is invalid. The candidate reported 'Decode Ways'; this is the standard version used for study.",
"Input:  '226'\nOutput: 3  # B B F; V F; B Z\n\nInput:  '06'\nOutput: 0",
"At each position, add ways from the previous position when the one-digit code is valid. Add ways from two positions back when the two-digit code is 10 through 26.",
"class Solution {\n  static long decodeWays(String s) {\n    if (s == null || s.isEmpty() || s.charAt(0) == '0')\n      return 0;\n    long prev2 = 1, prev1 = 1;\n    for (int i = 1; i < s.length(); i++) {\n      long current = 0;\n      if (s.charAt(i) != '0') current += prev1;\n      int pair = (s.charAt(i - 1) - '0') * 10\n          + (s.charAt(i) - '0');\n      if (pair >= 10 && pair <= 26) current += prev2;\n      prev2 = prev1;\n      prev1 = current;\n    }\n    return prev1;\n  }\n}",
"O(n) time and O(1) extra space.",
"Try '0', '101', '100', '27', and '11106'.")

problem(7,"Leader elements in an array","Reported topic - 2025; reconstructed prompt","FastPrep summary of a November 2025 AASE campus report [S5]",
"Return every element that is strictly greater than all elements to its right. The rightmost element counts as a leader. The source report is ambiguous about whether it asked for one leader or all leaders, so this exercise uses the standard all-leaders form.",
"Input:  [16, 17, 4, 3, 5, 2]\nOutput: [17, 5, 2]",
"Scan right to left and track the maximum seen. An element is a strict leader when it exceeds that maximum. Reverse the collected result.",
"import java.util.*;\nclass Solution {\n  static List<Integer> leaders(int[] a) {\n    List<Integer> found = new ArrayList<>();\n    long maximum = Long.MIN_VALUE;\n    for (int i = a.length - 1; i >= 0; i--) {\n      if (a[i] > maximum) {\n        found.add(a[i]);\n        maximum = a[i];\n      }\n    }\n    Collections.reverse(found);\n    return found;\n  }\n}",
"O(n) time and O(k) output space, where k is the number of leaders.",
"If your test says 'greater than or equal', change > to >=. Always follow the actual prompt.")

problem(8,"Longest subarray with sum at most K","Practice version of reported sliding-window topic","GeeksforGeeks AASE 2026-batch campus report [S6]",
"Given an array of nonnegative integers and K, return the maximum length of a contiguous subarray whose sum is at most K. The report named a dynamic sliding-window problem but did not provide its exact statement; this is original practice.",
"Input:  a=[2, 1, 3, 2, 1], K=5\nOutput: 2  # [2, 1] and [3, 2] are valid length-2 windows",
"A moving window works because all values are nonnegative. Extend right; shrink from left while the sum is too large. A subarray must be contiguous.",
"class Solution {\n  static int longestAtMostK(int[] a, long k) {\n    int left = 0, best = 0;\n    long sum = 0;\n    for (int right = 0; right < a.length; right++) {\n      sum += a[right];\n      while (sum > k && left <= right) {\n        sum -= a[left++];\n      }\n      best = Math.max(best, right - left + 1);\n    }\n    return best;\n  }\n}",
"O(n) time and O(1) extra space. This method needs nonnegative elements.",
"Try [0,0,0] with K=0. If negative numbers are allowed, this window rule is not sufficient.")

problem(9,"Fetch matching rows with a two-table join","Reported SQL topic; original practice schema","Naukri Code360 December 2025 report [S3]",
"Tables: Students(student_id, name, department_id) and Departments(department_id, department_name). Return each student's name and department name. Include only students with a matching department. Write standard SQL.",
"Students: (1, 'Asha', 10), (2, 'Ravi', 20)\nDepartments: (10, 'IT'), (20, 'ECE')\nResult: ('Asha','IT'), ('Ravi','ECE')",
"Identify the foreign-key relationship first. An INNER JOIN keeps rows that match in both tables. Qualify shared column names with aliases.",
"SELECT s.name, d.department_name\nFROM Students AS s\nINNER JOIN Departments AS d\n    ON s.department_id = d.department_id\nORDER BY s.student_id;",
"Typically O(rows) with an indexed join key; exact cost depends on the database engine and indexes.",
"What changes if the prompt asks to include students with no department? Use LEFT JOIN and expect NULL for department_name.", "SQL")

problem(10,"Three-table SQL join","Reported SQL topic; original practice schema","Naukri Code360 October 2025 and FastPrep report [S4, S5]",
"Tables: Products(product_id, product_name, category_id), Categories(category_id, category_name), Inventory(product_id, hub_name, quantity). Return the name, category, and quantity for products stocked at the 'Central' hub with quantity > 0.",
"Product: (7,'Battery',2); Category: (2,'Parts')\nInventory: (7,'Central',12)\nResult: ('Battery','Parts',12)",
"Join through the keys shown in the schema, then filter the hub and quantity. The source mentioned a three-table/product transit-hub task but did not disclose complete columns; this schema is invented for practice.",
"SELECT p.product_name, c.category_name, i.quantity\nFROM Products AS p\nJOIN Categories AS c\n    ON p.category_id = c.category_id\nJOIN Inventory AS i\n    ON p.product_id = i.product_id\nWHERE i.hub_name = 'Central'\n  AND i.quantity > 0\nORDER BY p.product_name;",
"Join cost depends on indexes and table sizes; filtering and keyed joins are the important correctness points here.",
"If a product has two Central inventory rows, this returns two rows. Aggregate with SUM(quantity) only if the prompt asks for one total per product.", "SQL")

problem(11,"Group and filter order totals","Original practice","Current campus reports mention GROUP BY and conditional SQL [S5]",
"Table: Orders(order_id, customer_id, amount). Return customer_id and total amount for customers whose summed amount is at least 1000. Sort by total descending.",
"Orders: (1,10,600), (2,10,500), (3,11,300)\nResult: (10,1100)",
"WHERE filters rows before grouping. HAVING filters grouped results after SUM. This distinction is common in technical and SQL rounds.",
"SELECT customer_id, SUM(amount) AS total_amount\nFROM Orders\nGROUP BY customer_id\nHAVING SUM(amount) >= 1000\nORDER BY total_amount DESC, customer_id;",
"A table scan is at least O(n); grouping and sorting cost depend on the database implementation.",
"Why is WHERE SUM(amount) >= 1000 invalid? Aggregate conditions belong in HAVING.", "SQL")

problem(12,"Button changes a message","Reported frontend task type; original practice","Naukri Code360 October 2025 candidate report [S4]",
"Build a small page with a button labeled 'Show message'. On click, change the paragraph text to 'Ready for the next round!' and give it a teal color. This is an original version of the reported DOM/event task.",
"Before click: 'Waiting...'\nAfter click:  'Ready for the next round!'",
"Select the elements, then attach a click listener. Keep text in textContent so it is handled as text rather than parsed HTML.",
"<!doctype html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"utf-8\">\n  <style>\n    body { font: 16px Arial; padding: 24px; }\n    button { padding: 8px 12px; cursor: pointer; }\n    .ready { color: #0b8f87; font-weight: bold; }\n  </style>\n</head>\n<body>\n  <p id=\"status\">Waiting...</p>\n  <button id=\"show\" type=\"button\">Show message</button>\n  <script>\n    const status = document.querySelector('#status');\n    document.querySelector('#show').addEventListener('click', () => {\n      status.textContent = 'Ready for the next round!';\n      status.classList.add('ready');\n    });\n  </script>\n</body>\n</html>",
"DOM lookup and one text update are effectively O(1) for this small page.",
"Click twice. The result should stay correct. Verify the button type does not accidentally submit a form.", "HTML / CSS / JavaScript")

problem(13,"Validate an email form","Original frontend practice","Current campus reports describe edits to provided HTML/CSS/JS [S4, S5]",
"Create a compact form with an email field and a submit button. Show 'Enter a valid email' for an invalid value, otherwise show 'Submitted'. This is original practice for modifying a supplied UI.",
"Input:  'student@example.com' -> 'Submitted'\nInput:  'not-an-email'        -> 'Enter a valid email'",
"Use the browser's built-in email validity check. Prevent default submission so the feedback remains visible.",
"<!doctype html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"utf-8\">\n  <style>\n    body { font: 16px Arial; padding: 24px; }\n    form { display: flex; gap: 8px; flex-wrap: wrap; }\n    input, button { padding: 8px; }\n  </style>\n</head>\n<body>\n  <form id=\"signup\" novalidate>\n    <label>Email <input id=\"email\" type=\"email\" required></label>\n    <button type=\"submit\">Submit</button>\n  </form>\n  <p id=\"result\" role=\"status\"></p>\n  <script>\n    const form = document.querySelector('#signup');\n    const email = document.querySelector('#email');\n    const result = document.querySelector('#result');\n    form.addEventListener('submit', (event) => {\n      event.preventDefault();\n      result.textContent = email.checkValidity()\n        ? 'Submitted' : 'Enter a valid email';\n    });\n  </script>\n</body>\n</html>",
"Constant work per submission for ordinary field lengths.",
"Try empty input, missing @, and a valid address. Keep accessibility labels and status text.", "HTML / CSS / JavaScript")

problem(14,"Two Sum","Original DSA practice","A September 2026 candidate described a basic hash-map task [S7]",
"Given integers a and target, return indices of two different elements whose sum equals target. Return an empty array when no pair exists. This is a standard practice task, not a confirmed exact Accenture prompt.",
"Input:  a=[2, 7, 11, 15], target=9\nOutput: (0, 1)",
"Store values already seen with their indices. Before storing the current value, check whether target - value is already present.",
"import java.util.*;\nclass Solution {\n  static int[] twoSum(int[] a, int target) {\n    Map<Integer, Integer> seen = new HashMap<>();\n    for (int i = 0; i < a.length; i++) {\n      long need = (long) target - a[i];\n      if (need >= Integer.MIN_VALUE\n          && need <= Integer.MAX_VALUE\n          && seen.containsKey((int) need))\n        return new int[] {seen.get((int) need), i};\n      seen.put(a[i], i);\n    }\n    return new int[0];\n  }\n}",
"O(n) expected time and O(n) extra space.",
"Try negative values and duplicate values. Do not use the same array element twice.")

section("60-minute three-part mock")
add("Attempt this page before looking at the answer outline. Use one uninterrupted 60-minute sitting: DSA 25 minutes, SQL 15 minutes, frontend 20 minutes.")
add("A. DSA - first non-repeating character", "H2X")
add("Given a string of lowercase letters, return its first character that occurs exactly once. Return an empty string if none exists. Example: 'swiss' -> 'w'.")
add("B. SQL - customer purchase count", "H2X")
add("Tables Customers(customer_id, name) and Orders(order_id, customer_id). Return every customer's name and number of orders, including customers with zero orders. Sort by name.")
add("C. Frontend - counter", "H2X")
add("Create a displayed count starting at 0 and a button that increments the count by 1 each time it is clicked. Use a click listener and textContent.")
add("Self-score", "H2X")
bullet("DSA: correct frequency count, first-in-original-order selection, empty/no-answer cases.")
bullet("SQL: LEFT JOIN, COUNT(o.order_id) rather than COUNT(*), GROUP BY, ordering.")
bullet("Frontend: visible initial state, listener, numeric increment, repeated clicks.")

section("Mock answers")
add("A. DSA answer", "H2X")
code("class Solution {\n  static String firstUnique(String s) {\n    int[] freq = new int[26];\n    for (int i = 0; i < s.length(); i++)\n      freq[s.charAt(i) - 'a']++;\n    for (int i = 0; i < s.length(); i++) {\n      if (freq[s.charAt(i) - 'a'] == 1)\n        return String.valueOf(s.charAt(i));\n    }\n    return \"\";\n  }\n}")
add("O(n) time, O(k) space for k distinct letters (at most 26 under the stated constraint).")
add("B. SQL answer", "H2X")
code("SELECT c.name, COUNT(o.order_id) AS order_count\nFROM Customers AS c\nLEFT JOIN Orders AS o\n    ON c.customer_id = o.customer_id\nGROUP BY c.customer_id, c.name\nORDER BY c.name;")
add("COUNT(o.order_id) returns 0 for a customer with no matching order; COUNT(*) would return 1 for that LEFT JOIN row.")
add("C. Frontend answer", "H2X")
code("<p id=\"count\">0</p>\n<button id=\"add\" type=\"button\">Add 1</button>\n<script>\n  let value = 0;\n  const display = document.querySelector('#count');\n  document.querySelector('#add').addEventListener('click', () => {\n    value += 1;\n    display.textContent = String(value);\n  });\n</script>")

section("Four-week use plan")
rows=[
    [P("Week", "THead"),P("Guide work", "THead"),P("Target", "THead")],
    [P("1", "TCell"),P("Problems 1-5; rewrite each without looking", "TCell"),P("Solve each within 15 min", "TCell")],
    [P("2", "TCell"),P("Problems 6-8 and 14; learn patterns", "TCell"),P("Explain code and edge cases aloud", "TCell")],
    [P("3", "TCell"),P("Problems 9-13; type SQL and web code yourself", "TCell"),P("Finish a DSA/SQL/web set in 60 min", "TCell")],
    [P("4", "TCell"),P("Repeat mock twice; revisit all mistakes", "TCell"),P("Consistent clean solutions under time", "TCell")],
]
t=Table(rows,colWidths=[18*mm,91*mm,65*mm])
t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),("VALIGN",(0,0),(-1,-1),"TOP"),("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#D9E3ED")),("INNERGRID",(0,0),(-1,-1),0.3,colors.HexColor("#D9E3ED")),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
story.append(t)
add("After every question", "H2X")
bullet("Write down the pattern, the bug you made, and the edge case that caught it.")
bullet("Re-solve missed questions after 2 days and 7 days without notes.")
bullet("Spend separate time on cognitive games, technical MCQs, communication, projects, and interview answers.")

section("Sources and accuracy notes")
add("These links support the reported topics and broad assessment pattern. They do not authenticate any exact future paper. The answer code, examples, and practice questions in this guide are original educational material.")
sources=[
    ("S1", "2023 on-campus: next smaller element and string casing", "https://www.geeksforgeeks.org/interview-experiences/accenture-coding-questions-2023-23rd-july-on-campus-hiring/"),
    ("S2", "2024 assessment: prime number and string formatting", "https://www.geeksforgeeks.org/interview-experiences/accenture-interview-experience-2025-on-campus/"),
    ("S3", "Dec 2025 campus: character count and SQL joins", "https://www.naukri.com/code360/interview-experiences/accenture-india-pvt-ltd/interview-experience-by-vaibhav-bhatt-on-campus-dec-2025"),
    ("S4", "Oct 2025 campus: Decode Ways, SQL, and HTML/CSS/JS", "https://www.naukri.com/code360/interview-experiences/accenture-solutions-private-limited/accenture-interview-experience-on-campus-oct-2025"),
    ("S5", "2025-2026 campus report synthesis: leader, joins, web, variants", "https://www.fastprep.io/experience/accenture-new-grad-sde-oa-interviews-ase-aase-aae-aeh-2025-2026-current-experien-x8g4q1maot"),
    ("S6", "AASE 2026-batch campus: dynamic sliding window, SQL, frontend", "https://www.geeksforgeeks.org/interview-experiences/accenture-interview-experience-for-aase-advanced-associate-software-engineer-2026-batch-on-campus/"),
    ("S7", "Sep 2026 candidate report: hash-map DSA, SQL join, web task", "https://www.reddit.com/r/PlacementsPrep/comments/1wgbe2w/accenture_oa_experience_10th_sept_shift_2/"),
    ("S8", "Accenture official assessment guidance and AI rule", "https://www.accenture.com/in-en/careers/explore-careers/area-of-interest/journey-to-accenture"),
]
for key,title,url in sources:
    add(f"<link href='{escape(url)}' color='#2467A6'><b>[{key}] {escape(title)}</b></link>")
    add(f"Linked source: {escape(url.split('/')[2])}", "SmallX")
add("Final reminder", "H2X")
add("Check your placement-cell notice and assessment invitation for the actual order, language choices, time, permitted tools, and question format. Accenture says external AI use is not allowed during its technical assessment [S8]. Practice independently so you can reproduce these solutions yourself.")

doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=17*mm,leftMargin=17*mm,topMargin=20*mm,bottomMargin=19*mm,title="Accenture Campus Coding Questions - Java Edition",author="Codex",subject="Candidate-reported previous-year questions and original worked practice in Java")
doc.build(story,onFirstPage=on_page,onLaterPages=on_page)
print(OUT)
