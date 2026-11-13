import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const outputDir = path.join(root, "outputs", "battery_pinn_report");
const parseCsv = text => text.trim().split(/\r?\n/).map(line => line.split(",").map(value => {
  if (value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) && value.trim() !== "" ? numeric : value;
}));
const dataset = parseCsv(await fs.readFile(path.join(root, "data", "exports", "paper_dataset_excel.csv"), "utf8"));
const conditions = parseCsv(await fs.readFile(path.join(root, "data", "exports", "conditions_excel.csv"), "utf8"));
const metrics = parseCsv(await fs.readFile(path.join(root, "outputs", "paper_v1", "metrics.csv"), "utf8"));

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const conditionSheet = workbook.worksheets.add("Conditions");
const dataSheet = workbook.worksheets.add("Dataset");
const metricSheet = workbook.worksheets.add("Metrics");
const dictionary = workbook.worksheets.add("Data Dictionary");
const navy = "#17324D", orange = "#E76520", pale = "#EAF0F5", white = "#FFFFFF", ink = "#17212B", muted = "#5C6B78";

function title(sheet, range, text, subtitle) {
  const first = range.split(":")[0];
  sheet.getRange(first).values = [[text]];
  sheet.getRange(first).format = {fill: navy, font: {bold: true, color: white, size: 20}, rowHeight: 34, verticalAlignment: "center"};
  if (subtitle) {
    const row = Number(first.match(/\d+/)[0]) + 1;
    sheet.getRange(`A${row}`).values = [[subtitle]];
    sheet.getRange(`A${row}`).format = {font: {italic: true, color: muted}, wrapText: true};
  }
}
function header(range) {
  range.format = {fill: navy, font: {bold: true, color: white}, borders: {preset: "all", style: "thin", color: "#C7D2DC"}, verticalAlignment: "center", wrapText: true};
}
function writeChunked(sheet, rows, startRow=0, chunk=4000) {
  const cols = rows[0].length;
  for (let i=0;i<rows.length;i+=chunk) sheet.getRangeByIndexes(startRow+i,0,Math.min(chunk,rows.length-i),cols).values = rows.slice(i,i+chunk);
}

summary.showGridLines = false;
title(summary, "A1:H1", "PC-MP-PINN", "Battery dataset and experiment report. Quick results are not publication-final.");
summary.getRange("A4:A9").values = [["Dataset rows"],["Operating conditions"],["Training conditions"],["Validation conditions"],["Test conditions"],["Experiment runs"]];
summary.getRange("B4:B9").formulas = [["=ROWS(Dataset!A2:A45001)"],["=ROWS(Conditions!A2:A19)"],["=COUNTIF(Conditions!B2:B19,\"train\")"],["=COUNTIF(Conditions!B2:B19,\"validation\")"],["=COUNTIF(Conditions!B2:B19,\"test\")"],["=COUNTA(Metrics!A2:A487)/18"]];
summary.getRange("A4:A9").format = {fill:pale,font:{bold:true,color:ink}};
summary.getRange("B4:B9").format = {font:{bold:true,color:orange,size:16},numberFormat:"#,##0"};
summary.getRange("D4:H4").values = [["Model","Budget","Conc. rel-L2","Temp RMSE (K)","Stress RMSE (MPa)"]];
summary.getRange("D5:H7").values = [["Data-only",150,null,null,null],["Prior PINN",150,null,null,null],["PC-MP-PINN",150,null,null,null]];
summary.getRange("F5").formulas = [["=AVERAGEIFS(Metrics!$H$2:$H$487,Metrics!$B$2:$B$487,\"data_only\",Metrics!$C$2:$C$487,150,Metrics!$F$2:$F$487,\"test\",Metrics!$G$2:$G$487,\"concentration_relative_l2\")"]];
summary.getRange("G5").formulas = [["=AVERAGEIFS(Metrics!$H$2:$H$487,Metrics!$B$2:$B$487,\"data_only\",Metrics!$C$2:$C$487,150,Metrics!$F$2:$F$487,\"test\",Metrics!$G$2:$G$487,\"delta_temperature_rmse_K\")"]];
summary.getRange("H5").formulas = [["=AVERAGEIFS(Metrics!$H$2:$H$487,Metrics!$B$2:$B$487,\"data_only\",Metrics!$C$2:$C$487,150,Metrics!$F$2:$F$487,\"test\",Metrics!$G$2:$G$487,\"tangential_stress_rmse_MPa\")"]];
for (const [row,variant] of [[6,"prior_pinn"],[7,"pc_mp_pinn"]]) {
  summary.getRange(`F${row}`).formulas = [[`=AVERAGEIFS(Metrics!$H$2:$H$487,Metrics!$B$2:$B$487,"${variant}",Metrics!$C$2:$C$487,150,Metrics!$F$2:$F$487,"test",Metrics!$G$2:$G$487,"concentration_relative_l2")`]];
  summary.getRange(`G${row}`).formulas = [[`=AVERAGEIFS(Metrics!$H$2:$H$487,Metrics!$B$2:$B$487,"${variant}",Metrics!$C$2:$C$487,150,Metrics!$F$2:$F$487,"test",Metrics!$G$2:$G$487,"delta_temperature_rmse_K")`]];
  summary.getRange(`H${row}`).formulas = [[`=AVERAGEIFS(Metrics!$H$2:$H$487,Metrics!$B$2:$B$487,"${variant}",Metrics!$C$2:$C$487,150,Metrics!$F$2:$F$487,"test",Metrics!$G$2:$G$487,"tangential_stress_rmse_MPa")`]];
}
header(summary.getRange("D4:H4")); summary.getRange("F5:H7").format.numberFormat="0.0000";
summary.getRange("A12:H15").values = [["How to use this workbook",null,null,null,null,null,null,null],["Dataset","Filter the Dataset sheet by split, condition, time or radius. All numeric values retain SI units unless the header states MPa.",null,null,null,null,null,null],["Metrics","Filter the Metrics sheet by execution mode. Only full-mode results belong in the final paper table.",null,null,null,null,null,null],["Database","The companion SQLite file contains samples, conditions, experiment_runs and metrics tables.",null,null,null,null,null,null]];
summary.getRange("A12:H12").format={fill:orange,font:{bold:true,color:white}}; summary.getRange("A13:H15").format.wrapText=true; summary.getRange("13:15").format.rowHeight=40;
summary.getRange("A:H").format.columnWidth = 18; summary.getRange("A:A").format.columnWidth=24; summary.getRange("B:B").format.columnWidth=42; summary.getRange("D:D").format.columnWidth=20;

writeChunked(conditionSheet, conditions); header(conditionSheet.getRange(`A1:${String.fromCharCode(64+conditions[0].length)}1`)); conditionSheet.freezePanes.freezeRows(1); conditionSheet.tables.add(`A1:${String.fromCharCode(64+conditions[0].length)}${conditions.length}`,true,"ConditionsTable"); conditionSheet.getUsedRange().format.autofitColumns();
writeChunked(dataSheet, dataset); header(dataSheet.getRange(`A1:N1`)); dataSheet.freezePanes.freezeRows(1); dataSheet.freezePanes.freezeColumns(2); dataSheet.tables.add(`A1:N${dataset.length}`,true,"BatteryDatasetTable"); dataSheet.getRange("C2:N45001").format.numberFormat="0.0000"; dataSheet.getRange("A:N").format.columnWidth=17; dataSheet.getRange("A:B").format.columnWidth=14;
writeChunked(metricSheet, metrics); header(metricSheet.getRange("A1:H1")); metricSheet.freezePanes.freezeRows(1); metricSheet.tables.add(`A1:H${metrics.length}`,true,"MetricsTable"); metricSheet.getRange(`H2:H${metrics.length}`).format.numberFormat="0.000000"; metricSheet.getUsedRange().format.autofitColumns();

const definitions = [["Field","Unit","Meaning"],["condition_id","identifier","Unique operating condition C01-C18"],["split","category","Condition-level train, validation or test assignment"],["c_rate","C","Constant-current discharge rate"],["ambient_C","°C","Ambient and initial cell temperature"],["time_s","s","Physical simulation time"],["time_star","1","Time divided by the fixed 7200 s horizon"],["radius_star","1","Particle radius divided by particle radius"],["concentration_mol_m3","mol m^-3","Negative-particle lithium concentration"],["temperature_K","K","Volume-averaged cell temperature"],["voltage_V","V","Terminal voltage from PyBaMM"],["heat_W_m3","W m^-3","Volume-averaged total heat generation"],["surface_gradient_star","1","Dimensionless concentration gradient at the surface"],["sigma_radial_MPa","MPa","Analytical radial particle stress"],["sigma_tangential_MPa","MPa","Analytical tangential particle stress"]];
writeChunked(dictionary, definitions); header(dictionary.getRange("A1:C1")); dictionary.getRange("A:C").format.columnWidth=28; dictionary.getRange("C:C").format.columnWidth=52; dictionary.getUsedRange().format.wrapText=true; dictionary.freezePanes.freezeRows(1);

workbook.recalculate();
console.log((await workbook.inspect({kind:"table",range:"Summary!A1:H15",include:"values,formulas",tableMaxRows:15,tableMaxCols:8})).ndjson);
console.log((await workbook.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",options:{useRegex:true,maxResults:300},summary:"final formula error scan"})).ndjson);
await fs.mkdir(path.join(outputDir,"previews"),{recursive:true});
for (const [name,range] of [["Summary","A1:H15"],["Conditions","A1:G20"],["Dataset","A1:N28"],["Metrics","A1:H28"],["Data Dictionary","A1:C16"]]) {
  const preview = await workbook.render({sheetName:name,range,scale:1.2,format:"png"});
  await fs.writeFile(path.join(outputDir,"previews",`${name.replaceAll(" ","_")}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(outputDir,"battery_pinn_dataset.xlsx"));
console.log(path.join(outputDir,"battery_pinn_dataset.xlsx"));
