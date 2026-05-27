import { NextResponse } from "next/server";
import { exec } from "child_process";
import path from "path";
import fs from "fs";

const projectRoot = path.resolve(process.cwd(), "..");
const tempImagePath = path.join(projectRoot, "output", "temp_predict.png");

export async function POST(req: Request) {
  try {
    const { correctLabel } = await req.json();

    if (correctLabel === undefined || correctLabel === null) {
      return NextResponse.json(
        { error: "Thiếu nhãn đúng thực tế." },
        { status: 400 }
      );
    }

    if (!fs.existsSync(tempImagePath)) {
      return NextResponse.json(
        { error: "Không tìm thấy ảnh vẽ tay tạm thời để huấn luyện bổ sung." },
        { status: 400 }
      );
    }

    const cmd = `python3 finetune.py "${tempImagePath}" ${correctLabel}`;

    return new Promise<NextResponse>((resolve) => {
      exec(cmd, { cwd: projectRoot }, (error, stdout, stderr) => {
        if (error) {
          console.error("Finetune exec error:", error);
          return resolve(
            NextResponse.json(
              { error: `Lỗi khi tinh chỉnh mô hình: ${stderr || error.message}` },
              { status: 500 }
            )
          );
        }

        const outputText = stdout.toString();
        
        let beforePred = null;
        let beforeConf = null;
        let beforeLoss = null;
        let afterPred = null;
        let afterConf = null;
        let afterLoss = null;

        const beforePredMatch = outputText.match(/BEFORE_PRED:(\d)/);
        if (beforePredMatch) beforePred = parseInt(beforePredMatch[1]);

        const beforeConfMatch = outputText.match(/BEFORE_CONF:([\d.]+)/);
        if (beforeConfMatch) beforeConf = parseFloat(beforeConfMatch[1]);

        const beforeLossMatch = outputText.match(/BEFORE_LOSS:([\d.]+)/);
        if (beforeLossMatch) beforeLoss = parseFloat(beforeLossMatch[1]);

        const afterPredMatch = outputText.match(/AFTER_PRED:(\d)/);
        if (afterPredMatch) afterPred = parseInt(afterPredMatch[1]);

        const afterConfMatch = outputText.match(/AFTER_CONF:([\d.]+)/);
        if (afterConfMatch) afterConf = parseFloat(afterConfMatch[1]);

        const afterLossMatch = outputText.match(/AFTER_LOSS:([\d.]+)/);
        if (afterLossMatch) afterLoss = parseFloat(afterLossMatch[1]);

        resolve(
          NextResponse.json({
            success: true,
            before: { prediction: beforePred, confidence: beforeConf, loss: beforeLoss },
            after: { prediction: afterPred, confidence: afterConf, loss: afterLoss },
            rawOutput: outputText,
          })
        );
      });
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: `Lỗi xử lý yêu cầu: ${error.message}` },
      { status: 500 }
    );
  }
}
