const fs = require("fs")
const webdriver = require("selenium-webdriver")
const { PNG } = require("pngjs")
const Jimp = require('jimp')
const { exit } = require("process");

let url_list = fs.readFileSync("./url_list.json")
url_list = JSON.parse(url_list)

let start_time = 0;

function TimerReset() {
  // Reset Timer.
  console.log("[Time] Timer Reset.")
  start_time = new Date();
}

function TimerReport() {
  // Print out timer info.
  if (start_time == 0) {
    console.log("[Time] Timer is not setup yet.")
    return new Date(0);
  }
  else {
    let now = new Date();
    let interval = new Date(now.getTime() - start_time.getTime());
    console.log("[Time]", interval.getMinutes(), "m", interval.getSeconds(), "s.", interval.getMilliseconds())
    return interval;
  }
}

function Mat2Base64(mat) {
  /*
    Function: Convert Mat of OpenCv to 
 
    Parameter:
      mat: cv.Mat(), Image to be converted.
 
    Return: String, Image in base64.
  */
  const size = mat.size()
  const png = new PNG({ width: size.width, height: size.height })
  png.data.set(mat.data)
  const buffer = PNG.sync.write(png)
  return 'data:image/png;base64,' + buffer.toString('base64')
}

async function Screenshot(driver, name = "default") {
  /*
    Function: Taking screenshot and store in ./Data/...
 
    Parameter:
      driver: webdriver,  Driver create by selenium.
      name:   String,     File name
 
    Return: String, Image in base64.
  */
  let base64Data
  await driver.takeScreenshot().then(function (data) {
    base64Data = data.replace(/^data:image\/png;base64,/, "")
    // fs.writeFile("./Data/" + name + ".png", base64Data, 'base64', (err) => { if (err) console.log("Error on Screenshot png"); })
    // fs.writeFile("./Data/" + name + ".txt", base64Data, (err) => { if (err) console.log("Error on Screenshot txt"); })
  })
  return base64Data
}

async function Waiting(time) {
  /*
    Function: Use "await Waiting(t)" to wait t second. Can only used in async function.
    
    Param:
      time: Integer, Wait how much second.
  */
  for (let t = 0; t < time; ++t)
    for (let i = 0; i < 40000; ++i)
      for (let j = 0; j < 40000; ++j)
        continue;
}

function saveImg(img, name = "./Data/test.png") {
  /*
    Function: Store image to local.
 
    Param:
      img:  cv.Mat, Image to be stored.
      name: string, local position.
  */
  let str = Mat2Base64(img);
  str = str.replace(/^data:image\/png;base64,/, "")
  fs.writeFile(name, str, 'base64', (err) => { if (err) console.log("Error on saveImg()"); })
}

function Crop(img, x, y, width, height) {
  /*
    Function: Crop image.
 
    Param:
      img:    cv.Mat,   Image to be cropped.
      x, y:   Integer,  Left-Top position to be cropped.
      width:  Integer,  Width of result image.
      height:  Integer,  Height of result image.
 
    Function: cv.Mat,   Crop result.
  */
  let dst = new cv.Mat();
  let rect = new cv.Rect(x, y, width, height);
  dst = img.roi(rect).clone();
  return dst
}

function HistCheck(img, thres = 90) {
  /*
    Function: Checking the image is most of single color.
 
    Parameter:
      img:    cv.Mat(), A image.
      thres:  Number,   Threshold
 
    Return: Boolean, If any one of bins is over thres%, return true. Otherwise, return false.
  */
  let src = img.clone();
  cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
  let srcVec = new cv.MatVector();
  srcVec.push_back(src);
  let hist = new cv.Mat();
  let mask = new cv.Mat();
  // Param:   srcVec, channels, mask, hist, histSize, ranges, accumulate
  cv.calcHist(srcVec, [0], mask, hist, [16], [0, 256], false);
  let max = src.rows * src.cols;
  for (let i = 0; i < 16; i++) {
    let binVal = hist.data32F[i] * 100 / max;
    if (binVal >= thres) return true;
  }
  return false;
}

async function Base64_2_Mat(img_base64) {
  /*
    Function: Convert image from base64 to OpenCV's mat.
 
    Parameter:
      img_base64:     Screenshot generate by selenium's IDE
 
    Return: cv.Mat(), A image.
  */
  let imgBuf = Buffer.from(img_base64, 'base64')
  var jimpSrc = await Jimp.read(imgBuf)
  return cv.matFromImageData(jimpSrc.bitmap)
}

console.log("[INFO] loading OpenCV...")
async function onRuntimeInitialized() {
  // 用 try-catch 接 error 才不會狂噴錯誤訊息(openCV的關係)
  try {
    console.log("[INFO] Start Running AutoDiff...")

    // Need to check drivers' version !!
    // .chrome .firefox
    console.log("[INFO] Opening Drivers...")
    let driver1, driver2;

    driver1 = new webdriver.Builder().
      withCapabilities(webdriver.Capabilities.chrome()).
      build();
    driver2 = new webdriver.Builder().
      withCapabilities(webdriver.Capabilities.edge()).
      build();
    driver1.get("https://www.google.com.tw/")
    driver2.get("https://www.google.com.tw/")
    await driver1.manage().window().maximize();
    await driver2.manage().window().maximize();
    for (let i = 19; i < 68; ++i) {
      console.log("Running on case", i)
      driver1.get(url_list["list"][i])
      await driver2.get(url_list["list"][i])
      await Waiting(10)
      let img1_base64 = await Screenshot(driver1, "img1")
      let img2_base64 = await Screenshot(driver2, "img2")
      let src1 = await Base64_2_Mat(img1_base64)
      let src2 = await Base64_2_Mat(img2_base64)
      let W = Math.min(src1.size().width, src2.size().width) - 20 // minus scroll bar
      let H = Math.min(src1.size().height, src2.size().height)
      src1 = Crop(src1, 0, 0, W, H)
      src2 = Crop(src2, 0, 0, W, H)
      saveImg(src1, "./NewData/image/" + i.toString() + "_C.png")
      saveImg(src2, "./NewData/image/" + i.toString() + "_E.png")
      src1.delete()
      src2.delete()
    }
    driver1.close();
    driver2.close();
  }

  catch (e) {
    if (e.message === undefined) console.log("[Error] Unrecognizable Error.")
    else {
      console.log("[Error] failed occur! See 'error_msg.txt'.")
      fs.writeFileSync("./error_msg.txt", e.stack, (err) => { if (err) console.log("[Error] Failed on writing error_log.") })
    }
  }
  // // try-catch for copy
  // try {
  // }
  // catch (err) { console.log(err.stack); exit();}
}

Module = {
  onRuntimeInitialized
}

cv = require('./opencv.js')
