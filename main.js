const { app, BrowserWindow, screen } = require('electron');

let mascotWindow;
let dashboardWindow;

function createWindows() {
    // ---------------------------------------------------
    // 1. สร้างหน้าต่าง Dashboard (หน้าต่างโปรแกรมหลัก)
    // ---------------------------------------------------
    dashboardWindow = new BrowserWindow({
        width: 900,
        height: 600,
        title: "Office Syndrome AI",
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });
    dashboardWindow.loadFile('index.html');

    // ถ้าผู้ใช้กดกากบาทปิดหน้าต่าง Dashboard ให้ปิดแอปทั้งหมดเลย
    dashboardWindow.on('closed', () => {
        app.quit();
    });

    // ---------------------------------------------------
    // 2. สร้างหน้าต่าง Mascot (ตัวละครลอยๆ มุมจอ)
    // ---------------------------------------------------
    const { width, height } = screen.getPrimaryDisplay().workAreaSize;
    const windowWidth = 150;
    const windowHeight = 150;

    mascotWindow = new BrowserWindow({
        width: windowWidth,
        height: windowHeight,
        x: width - windowWidth,
        y: height - windowHeight,
        transparent: true,
        frame: false,
        alwaysOnTop: true,
        resizable: false,
        skipTaskbar: true,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });

    mascotWindow.setIgnoreMouseEvents(true);
    mascotWindow.loadFile('mascot.html');
}

app.whenReady().then(() => {
    createWindows();
});

app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') app.quit();
});