import { NextResponse } from "next/server";
import { spawn } from "child_process";
import fs from "fs";
import path from "path";

// Trạng thái huấn luyện toàn cục (in-memory)
let isTraining = false;
let trainingLogs: string[] = [];
let trainProcess: any = null;

const projectRoot = path.resolve(process.cwd(), "..");
const logDir = path.join(projectRoot, "output", "logs");

function loadEnvConfig() {
  const envPath = path.join(projectRoot, ".env");
  if (fs.existsSync(envPath)) {
    try {
      const envContent = fs.readFileSync(envPath, "utf-8");
      envContent.split("\n").forEach((line) => {
        const trimmed = line.trim();
        if (trimmed && !trimmed.startsWith("#") && trimmed.includes("=")) {
          const parts = trimmed.split("=");
          const key = parts[0].trim();
          const val = parts.slice(1).join("=").trim();
          process.env[key] = val;
        }
      });
    } catch (e) {
      console.error("Error reading env config:", e);
    }
  }
}

export async function GET() {
  loadEnvConfig();

  // Tìm log file mới nhất để đọc nếu không có tiến trình đang chạy trực tiếp
  let latestLogContent = "";
  try {
    if (fs.existsSync(logDir)) {
      const files = fs.readdirSync(logDir)
        .filter(f => f.startsWith("training_") && f.endsWith(".log"))
        .sort();
      if (files.length > 0) {
        const latestFile = path.join(logDir, files[files.length - 1]);
        latestLogContent = fs.readFileSync(latestFile, "utf-8");
      }
    }
  } catch (error) {
    console.error("Error reading logs:", error);
  }

  const numTrain = parseInt(process.env.MNIST_NUM_TRAIN || "1000");
  const numTest = parseInt(process.env.MNIST_NUM_TEST || "1000");

  return NextResponse.json({
    isTraining,
    logs: isTraining ? trainingLogs : latestLogContent.split("\n"),
    numTrain,
    numTest,
  });
}


export async function POST() {
  loadEnvConfig();

  if (isTraining) {

    return NextResponse.json(
      { error: "Tiến trình huấn luyện đang chạy." },
      { status: 400 }
    );
  }

  isTraining = true;
  trainingLogs = ["🚀 Đang khởi động tiến trình huấn luyện CNN thuần NumPy...\n"];

  // Chạy file train.py bằng python3
  const pythonPath = "python3";
  const scriptPath = path.join(projectRoot, "train.py");

  try {
    trainProcess = spawn(pythonPath, [scriptPath], {
      cwd: projectRoot,
      env: { ...process.env, PYTHONUNBUFFERED: "1" } // Đảm bảo log được stream liên tục
    });

    trainProcess.stdout.on("data", (data: Buffer) => {
      const text = data.toString();
      trainingLogs.push(text);
      // Giới hạn số lượng log dòng trong memory để tránh tràn bộ nhớ
      if (trainingLogs.length > 1000) {
        trainingLogs.shift();
      }
    });

    trainProcess.stderr.on("data", (data: Buffer) => {
      const text = data.toString();
      trainingLogs.push(`⚠️ [ERROR] ${text}`);
    });

    trainProcess.on("close", (code: number) => {
      isTraining = false;
      trainingLogs.push(`\n🏁 Tiến trình kết thúc với mã thoát: ${code}`);
      if (code === 0) {
        trainingLogs.push("✅ Huấn luyện hoàn thành thành công!");
      } else {
        trainingLogs.push("❌ Huấn luyện thất bại, vui lòng kiểm tra lại log.");
      }
      trainProcess = null;
    });

    return NextResponse.json({ message: "Bắt đầu huấn luyện thành công." });
  } catch (error: any) {
    isTraining = false;
    return NextResponse.json(
      { error: `Không thể chạy train.py: ${error.message}` },
      { status: 500 }
    );
  }
}
