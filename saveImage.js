const fs = require("fs")
const webdriver = require("selenium-webdriver")
const { PNG } = require("pngjs")
const Jimp = require('jimp')

url = 'https://www.chinatimes.com/?chdtv'

function Mat2Base64(mat) {
    const size = mat.size()
    const png = new PNG({ width: size.width, height: size.height })
    png.data.set(mat.data)
    const buffer = PNG.sync.write(png)
    return 'data:image/png;base64,' + buffer.toString('base64')
}

async function Screenshot(driver, name = "default") {
    let base64Data
    await driver.takeScreenshot().then(function (data) {
        base64Data = data.replace(/^data:image\/png;base64,/, "")
    })
    return base64Data
}

async function Waiting(time) {
    for (let t = 0; t < time; ++t)
        for (let i = 0; i < 40000; ++i)
            for (let j = 0; j < 40000; ++j)
                continue;
}

function saveImg(img, name = "./Data/test.png") {
    let str = Mat2Base64(img);
    str = str.replace(/^data:image\/png;base64,/, "")
    fs.writeFile(name, str, 'base64', (err) => { if (err) console.log("Error on saveImg()"); })
}

function Crop(img, x, y, width, height) {
    let dst = new cv.Mat();
    let rect = new cv.Rect(x, y, width, height);
    dst = img.roi(rect).clone();
    return dst
}

async function Base64_2_Mat(img_base64) {
    let imgBuf = Buffer.from(img_base64, 'base64')
    var jimpSrc = await Jimp.read(imgBuf)
    return cv.matFromImageData(jimpSrc.bitmap)
}

console.log("[INFO] loading OpenCV...")
async function onRuntimeInitialized() {

    try {
        let driver1, driver2;
        driver1 = new webdriver.Builder().
            withCapabilities(webdriver.Capabilities.chrome()).
            build();
        driver2 = new webdriver.Builder().
            withCapabilities(webdriver.Capabilities.edge()).
            build();
        driver1.get(url)
        driver2.get(url)
        await driver1.manage().window().maximize();
        await driver2.manage().window().maximize();

        await Waiting(10)

        let img1_base64 = await Screenshot(driver1, "img1")
        let img2_base64 = await Screenshot(driver2, "img2")
        let src1 = await Base64_2_Mat(img1_base64)
        let src2 = await Base64_2_Mat(img2_base64)
        let W = Math.min(src1.size().width, src2.size().width) - 20 // minus scroll bar
        let H = Math.min(src1.size().height, src2.size().height)
        src1 = Crop(src1, 0, 0, W, H)
        src2 = Crop(src2, 0, 0, W, H)
        saveImg(src1, "./Data/test_img1.png")
        saveImg(src2, "./Data/test_img2.png")
        src1.delete()
        src2.delete()

        driver1.close();
        driver2.close();
    } catch (e) {
        if (e.message === undefined) console.log("[Error] Unrecognizable Error.")
        else {
            console.log("[Error] failed occur! See 'error_msg.txt'.")
            fs.writeFileSync("./error_msg.txt", e.stack, (err) => { if (err) console.log("[Error] Failed on writing error_log.") })
        }
    }
}

Module = {
    onRuntimeInitialized
}

cv = require('./opencv.js')
