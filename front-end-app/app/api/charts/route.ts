import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const projectRoot = path.resolve(process.cwd(), "..");
const chartsDir = path.join(projectRoot, "output", "charts");

export async function GET() {
  const chartNames = [
    "sample_digits.png",
    "pixel_matrix.png",
    "training_history.png",
    "predictions.png",
  ];

  const charts: Record<string, string> = {};

  for (const name of chartNames) {
    const filePath = path.join(chartsDir, name);
    if (fs.existsSync(filePath)) {
      try {
        const buffer = fs.readFileSync(filePath);
        charts[name.replace(".png", "")] = `data:image/png;base64,${buffer.toString("base64")}`;
      } catch (error) {
        console.error(`Error reading chart ${name}:`, error);
      }
    }
  }

  return NextResponse.json(charts);
}
