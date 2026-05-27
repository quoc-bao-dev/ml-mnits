import { NextResponse } from "next/server";
import { exec } from "child_process";
import fs from "fs";
import path from "path";

const projectRoot = path.resolve(process.cwd(), "..");
const visualDataPath = path.join(projectRoot, "output", "visual_logs", "visual_data.json");

let isRunning = false;

export async function GET() {
  // Trả về dữ liệu visual nếu có
  if (!fs.existsSync(visualDataPath)) {
    return NextResponse.json({ exists: false, isRunning });
  }

  try {
    const raw = fs.readFileSync(visualDataPath, "utf-8");
    const data = JSON.parse(raw);
    return NextResponse.json({ exists: true, isRunning, data });
  } catch (e: any) {
    return NextResponse.json(
      { exists: false, isRunning, error: e.message },
      { status: 500 }
    );
  }
}

export async function POST() {
  if (isRunning) {
    return NextResponse.json(
      { error: "Đang chạy trực quan hóa..." },
      { status: 400 }
    );
  }

  isRunning = true;

  const cmd = `python3 train_visual.py`;

  return new Promise<NextResponse>((resolve) => {
    exec(cmd, { cwd: projectRoot, timeout: 30000 }, (error, stdout, stderr) => {
      isRunning = false;

      if (error) {
        console.error("train_visual error:", error);
        return resolve(
          NextResponse.json(
            { error: `Lỗi: ${stderr || error.message}` },
            { status: 500 }
          )
        );
      }

      resolve(
        NextResponse.json({ success: true, output: stdout.toString() })
      );
    });
  });
}
