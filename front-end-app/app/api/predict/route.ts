import { NextResponse } from "next/server";
import { exec } from "child_process";
import fs from "fs";
import path from "path";

const projectRoot = path.resolve(process.cwd(), "..");
const tempImagePath = path.join(projectRoot, "output", "temp_predict.png");
const predictionChartPath = path.join(projectRoot, "output", "charts", "prediction_single.png");

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { image, useSample, action, trueLabel } = body;

    // 1. Trường hợp chỉ chọn ảnh mẫu ngẫu nhiên từ MNIST (chưa dự đoán)
    if (action === "select_sample") {
      const selectCmd = `python3 select_sample.py`;
      return new Promise<NextResponse>((resolve) => {
        exec(selectCmd, { cwd: projectRoot }, (error, stdout, stderr) => {
          if (error) {
            console.error("Select sample error:", error);
            return resolve(
              NextResponse.json(
                { error: `Lỗi khi chọn ảnh mẫu: ${stderr || error.message}` },
                { status: 500 }
              )
            );
          }

          const outputText = stdout.toString();
          let label = null;
          const labelMatch = outputText.match(/LABEL:(\d)/);
          if (labelMatch) label = parseInt(labelMatch[1]);

          // Đọc ảnh mẫu vừa được lưu ra file temp
          let inputBase64 = "";
          try {
            if (fs.existsSync(tempImagePath)) {
              const inputBuffer = fs.readFileSync(tempImagePath);
              inputBase64 = `data:image/png;base64,${inputBuffer.toString("base64")}`;
            }
          } catch (inputErr) {
            console.error("Read temp sample image error:", inputErr);
          }

          resolve(
            NextResponse.json({
              success: true,
              inputImage: inputBase64,
              trueLabel: label,
            })
          );
        });
      });
    }

    // 2. Trường hợp chạy dự đoán
    // Dù vẽ tay hay ảnh mẫu, ta đều chạy trên file temp_predict.png
    let cmd = `python3 predict.py "${tempImagePath}" --visualize`;

    if (!useSample && image) {
      // image là chuỗi base64 của canvas vẽ tay
      const base64Data = image.replace(/^data:image\/\w+;base64,/, "");
      const buffer = Buffer.from(base64Data, "base64");

      // Đảm bảo thư mục output tồn tại và ghi file
      fs.mkdirSync(path.dirname(tempImagePath), { recursive: true });
      fs.writeFileSync(tempImagePath, buffer);
    } else if (useSample && trueLabel !== undefined && trueLabel !== null) {
      // Nếu dùng ảnh mẫu MNIST đã được chọn trước, truyền nhãn thật vào CLI làm tham số thứ 2
      cmd = `python3 predict.py "${tempImagePath}" ${trueLabel} --visualize`;
    }


    return new Promise<NextResponse>((resolve) => {
      exec(cmd, { cwd: projectRoot }, (error, stdout, stderr) => {

        if (error) {
          console.error("Exec error:", error);
          return resolve(
            NextResponse.json(
              { error: `Lỗi khi chạy mô hình dự đoán: ${stderr || error.message}` },
              { status: 500 }
            )
          );
        }

        // Phân tích kết quả từ stdout
        // Nhờ predict.py in ra kết quả có cấu trúc, ta có thể regex tìm kiếm
        const outputText = stdout.toString();
        
        let prediction = null;
        let confidence = null;
        let trueLabel = null;
        const probabilities: number[] = [];

        // Trích xuất: Chữ số dự đoán : X
        const predMatch = outputText.match(/Chữ số dự đoán\s*:\s*(\d)/);
        if (predMatch) prediction = parseInt(predMatch[1]);

        // Trích xuất: Độ tin cậy     : X%
        const confMatch = outputText.match(/Độ tin cậy\s*:\s*([\d.]+)%/);
        if (confMatch) confidence = parseFloat(confMatch[1]) / 100;

        // Trích xuất: Nhãn thật      : X
        const trueMatch = outputText.match(/Nhãn thật\s*:\s*(\d)/);
        if (trueMatch) trueLabel = parseInt(trueMatch[1]);

        // Trích xuất: X: YY.Y%
        const lines = outputText.split("\n");
        for (const line of lines) {
          const probMatch = line.match(/^\s*(\d):\s*([\d.]+)%/);
          if (probMatch) {
            probabilities.push(parseFloat(probMatch[2]) / 100);
          }
        }

        // Đọc ảnh kết quả dự đoán và chuyển sang base64
        let chartBase64 = "";
        try {
          if (fs.existsSync(predictionChartPath)) {
            const chartBuffer = fs.readFileSync(predictionChartPath);
            chartBase64 = `data:image/png;base64,${chartBuffer.toString("base64")}`;
          }
        } catch (chartErr) {
          console.error("Read chart error:", chartErr);
        }

        // Đọc ảnh đầu vào (vẽ hoặc trích xuất ngẫu nhiên từ MNIST)
        let inputBase64 = "";
        try {
          if (fs.existsSync(tempImagePath)) {
            const inputBuffer = fs.readFileSync(tempImagePath);
            inputBase64 = `data:image/png;base64,${inputBuffer.toString("base64")}`;
          }
        } catch (inputErr) {
          console.error("Read input image error:", inputErr);
        }

        // Đọc dữ liệu visualize
        let visualData = null;
        const visualPathMatch = outputText.match(/VISUAL_DATA_PATH:(.*)/);
        if (visualPathMatch) {
            try {
                const visualPath = visualPathMatch[1].trim();
                if (fs.existsSync(visualPath)) {
                    const visualContent = fs.readFileSync(visualPath, 'utf8');
                    visualData = JSON.parse(visualContent);
                }
            } catch (err) {
                console.error("Parse visual data error:", err);
            }
        }

        resolve(
          NextResponse.json({
            success: true,
            prediction,
            confidence,
            trueLabel,
            probabilities: probabilities.length === 10 ? probabilities : null,
            chartImage: chartBase64,
            inputImage: inputBase64,
            visualData,
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
