# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Mainwindow(object):
    def setupUi(self, Mainwindow):
        Mainwindow.setObjectName("Mainwindow")
        Mainwindow.resize(1600, 800)
        Mainwindow.setMinimumSize(QtCore.QSize(1600, 800))
        self.frame = QtWidgets.QFrame(Mainwindow)
        self.frame.setGeometry(QtCore.QRect(290, 30, 1401, 700))
        self.frame.setMinimumSize(QtCore.QSize(1200, 700))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame_2 = QtWidgets.QFrame(Mainwindow)
        self.frame_2.setGeometry(QtCore.QRect(40, 40, 250, 661))
        self.frame_2.setMinimumSize(QtCore.QSize(250, 600))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.layoutWidget = QtWidgets.QWidget(self.frame_2)
        self.layoutWidget.setGeometry(QtCore.QRect(40, 400, 152, 164))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setMinimumSize(QtCore.QSize(130, 50))
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.pushButton_6 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_6.setMinimumSize(QtCore.QSize(150, 50))
        self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout_2.addWidget(self.pushButton_6)
        self.pushButton_7 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_7.setMinimumSize(QtCore.QSize(150, 50))
        self.pushButton_7.setObjectName("pushButton_7")
        self.verticalLayout_2.addWidget(self.pushButton_7)
        self.layoutWidget1 = QtWidgets.QWidget(self.frame_2)
        self.layoutWidget1.setGeometry(QtCore.QRect(40, 20, 152, 332))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget1)
        self.label.setMinimumSize(QtCore.QSize(130, 50))
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.pushButton_1 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_1.setMinimumSize(QtCore.QSize(150, 50))
        self.pushButton_1.setObjectName("pushButton_1")
        self.verticalLayout.addWidget(self.pushButton_1)
        self.pushButton_2 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_2.setMinimumSize(QtCore.QSize(150, 50))
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_3.setMinimumSize(QtCore.QSize(150, 50))
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_4.setMinimumSize(QtCore.QSize(150, 50))
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout.addWidget(self.pushButton_4)
        self.pushButton_5 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_5.setMinimumSize(QtCore.QSize(150, 50))
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout.addWidget(self.pushButton_5)

        self.retranslateUi(Mainwindow)
        QtCore.QMetaObject.connectSlotsByName(Mainwindow)

    def retranslateUi(self, Mainwindow):
        _translate = QtCore.QCoreApplication.translate
        Mainwindow.setWindowTitle(_translate("Mainwindow", "Form"))
        self.label_2.setText(_translate("Mainwindow", "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">图像处理:</span></p></body></html>"))
        self.pushButton_6.setText(_translate("Mainwindow", "检测"))
        self.pushButton_7.setText(_translate("Mainwindow", "图像修复"))
        self.label.setText(_translate("Mainwindow", "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">基本操作:</span></p></body></html>"))
        self.pushButton_1.setText(_translate("Mainwindow", "灰度/二值化"))
        self.pushButton_2.setText(_translate("Mainwindow", "图像翻转"))
        self.pushButton_3.setText(_translate("Mainwindow", "图像锐化"))
        self.pushButton_4.setText(_translate("Mainwindow", "加噪"))
        self.pushButton_5.setText(_translate("Mainwindow", "滤波"))

class Ui_Formwin1(object):
    def setupUi(self, Formwin1):
        Formwin1.setObjectName("Formwin1")
        Formwin1.resize(1100, 800)
        Formwin1.setMinimumSize(QtCore.QSize(1100, 800))
        self.layoutWidget = QtWidgets.QWidget(Formwin1)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 50, 1008, 311))
        self.layoutWidget.setMinimumSize(QtCore.QSize(500, 0))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_load = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_load.setMinimumSize(QtCore.QSize(400, 50))
        self.pushButton_load.setObjectName("pushButton_load")
        self.gridLayout.addWidget(self.pushButton_load, 1, 0, 1, 1)
        self.label_jieguo = QtWidgets.QLabel(self.layoutWidget)
        self.label_jieguo.setMinimumSize(QtCore.QSize(500, 0))
        self.label_jieguo.setObjectName("label_jieguo")
        self.gridLayout.addWidget(self.label_jieguo, 0, 1, 1, 1)
        self.label_daichuli = QtWidgets.QLabel(self.layoutWidget)
        self.label_daichuli.setMinimumSize(QtCore.QSize(500, 0))
        self.label_daichuli.setObjectName("label_daichuli")
        self.gridLayout.addWidget(self.label_daichuli, 0, 0, 1, 1)
        self.pushButton_save = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_save.setMinimumSize(QtCore.QSize(400, 50))
        self.pushButton_save.setObjectName("pushButton_save")
        self.gridLayout.addWidget(self.pushButton_save, 1, 1, 1, 1)
        self.layoutWidget1 = QtWidgets.QWidget(Formwin1)
        self.layoutWidget1.setGeometry(QtCore.QRect(180, 390, 661, 111))
        self.layoutWidget1.setMinimumSize(QtCore.QSize(80, 50))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_turntoGray = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_turntoGray.setMinimumSize(QtCore.QSize(80, 50))
        self.pushButton_turntoGray.setObjectName("pushButton_turntoGray")
        self.horizontalLayout.addWidget(self.pushButton_turntoGray)
        self.pushButton_turntotwo = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_turntotwo.setMinimumSize(QtCore.QSize(80, 50))
        self.pushButton_turntotwo.setObjectName("pushButton_turntotwo")
        self.horizontalLayout.addWidget(self.pushButton_turntotwo)

        self.retranslateUi(Formwin1)
        QtCore.QMetaObject.connectSlotsByName(Formwin1)

    def retranslateUi(self, Formwin1):
        _translate = QtCore.QCoreApplication.translate
        Formwin1.setWindowTitle(_translate("Formwin1", "Form"))
        self.pushButton_load.setText(_translate("Formwin1", "选择图片"))
        self.label_jieguo.setText(_translate("Formwin1", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">结果</span></p></body></html>"))
        self.label_daichuli.setText(_translate("Formwin1", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">待处理</span></p></body></html>"))
        self.pushButton_save.setText(_translate("Formwin1", "保存图片"))
        self.pushButton_turntoGray.setText(_translate("Formwin1", "转成灰度图"))
        self.pushButton_turntotwo.setText(_translate("Formwin1", "将图片二值化"))


class Ui_Formwin2(object):
    def setupUi(self, Formwin2):
        Formwin2.setObjectName("Formwin2")
        Formwin2.resize(1100, 800)
        Formwin2.setMinimumSize(QtCore.QSize(1100, 800))
        self.frame = QtWidgets.QFrame(Formwin2)
        self.frame.setGeometry(QtCore.QRect(20, 410, 811, 151))
        self.frame.setMinimumSize(QtCore.QSize(720, 0))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.layoutWidget = QtWidgets.QWidget(self.frame)
        self.layoutWidget.setGeometry(QtCore.QRect(50, 60, 721, 61))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_tips = QtWidgets.QLabel(self.layoutWidget)
        self.label_tips.setMinimumSize(QtCore.QSize(150, 50))
        self.label_tips.setObjectName("label_tips")
        self.horizontalLayout_2.addWidget(self.label_tips)
        self.pushButton_fun0 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_fun0.setMinimumSize(QtCore.QSize(150, 50))
        self.pushButton_fun0.setObjectName("pushButton_fun0")
        self.horizontalLayout_2.addWidget(self.pushButton_fun0)
        self.pushButton_fun1 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_fun1.setMinimumSize(QtCore.QSize(150, 50))
        self.pushButton_fun1.setObjectName("pushButton_fun1")
        self.horizontalLayout_2.addWidget(self.pushButton_fun1)
        self.pushButton_fun11 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_fun11.setMinimumSize(QtCore.QSize(150, 50))
        self.pushButton_fun11.setObjectName("pushButton_fun11")
        self.horizontalLayout_2.addWidget(self.pushButton_fun11)
        self.layoutWidget1 = QtWidgets.QWidget(Formwin2)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 20, 1010, 308))
        self.layoutWidget1.setMinimumSize(QtCore.QSize(500, 0))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_daichuli = QtWidgets.QLabel(self.layoutWidget1)
        self.label_daichuli.setMinimumSize(QtCore.QSize(500, 0))
        self.label_daichuli.setObjectName("label_daichuli")
        self.horizontalLayout.addWidget(self.label_daichuli)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.pushButton_load = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_load.setMinimumSize(QtCore.QSize(500, 30))
        self.pushButton_load.setObjectName("pushButton_load")
        self.gridLayout.addWidget(self.pushButton_load, 1, 0, 1, 1)
        self.pushButton_save = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_save.setMinimumSize(QtCore.QSize(500, 30))
        self.pushButton_save.setObjectName("pushButton_save")
        self.gridLayout.addWidget(self.pushButton_save, 1, 1, 1, 1)
        self.label_jieguo = QtWidgets.QLabel(self.layoutWidget1)
        self.label_jieguo.setMinimumSize(QtCore.QSize(500, 0))
        self.label_jieguo.setObjectName("label_jieguo")
        self.gridLayout.addWidget(self.label_jieguo, 0, 1, 1, 1)

        self.retranslateUi(Formwin2)
        QtCore.QMetaObject.connectSlotsByName(Formwin2)

    def retranslateUi(self, Formwin2):
        _translate = QtCore.QCoreApplication.translate
        Formwin2.setWindowTitle(_translate("Formwin2", "Form"))
        self.label_tips.setText(_translate("Formwin2", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">操作：</span></p></body></html>"))
        self.pushButton_fun0.setText(_translate("Formwin2", "水平翻转"))
        self.pushButton_fun1.setText(_translate("Formwin2", "垂直翻转"))
        self.pushButton_fun11.setText(_translate("Formwin2", "沿xy轴翻转"))
        self.label_daichuli.setText(_translate("Formwin2", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">待处理</span></p></body></html>"))
        self.pushButton_load.setText(_translate("Formwin2", "选择图片"))
        self.pushButton_save.setText(_translate("Formwin2", "保存图片"))
        self.label_jieguo.setText(_translate("Formwin2", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">结果</span></p></body></html>"))


class Ui_Formwin3(object):
    def setupUi(self, Formwin3):
        Formwin3.setObjectName("Formwin3")
        Formwin3.resize(1100, 800)
        Formwin3.setMinimumSize(QtCore.QSize(1100, 800))
        self.layoutWidget = QtWidgets.QWidget(Formwin3)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 50, 1008, 311))
        self.layoutWidget.setMinimumSize(QtCore.QSize(500, 0))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_load = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_load.setMinimumSize(QtCore.QSize(400, 50))
        self.pushButton_load.setObjectName("pushButton_load")
        self.gridLayout.addWidget(self.pushButton_load, 1, 0, 1, 1)
        self.label_jieguo = QtWidgets.QLabel(self.layoutWidget)
        self.label_jieguo.setMinimumSize(QtCore.QSize(500, 0))
        self.label_jieguo.setObjectName("label_jieguo")
        self.gridLayout.addWidget(self.label_jieguo, 0, 1, 1, 1)
        self.label_daichuli = QtWidgets.QLabel(self.layoutWidget)
        self.label_daichuli.setMinimumSize(QtCore.QSize(500, 0))
        self.label_daichuli.setObjectName("label_daichuli")
        self.gridLayout.addWidget(self.label_daichuli, 0, 0, 1, 1)
        self.pushButton_save = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_save.setMinimumSize(QtCore.QSize(400, 50))
        self.pushButton_save.setObjectName("pushButton_save")
        self.gridLayout.addWidget(self.pushButton_save, 1, 1, 1, 1)
        self.frame = QtWidgets.QFrame(Formwin3)
        self.frame.setGeometry(QtCore.QRect(604, 391, 16, 50))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.widget = QtWidgets.QWidget(Formwin3)
        self.widget.setGeometry(QtCore.QRect(280, 400, 471, 161))
        self.widget.setObjectName("widget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pushButton_lap = QtWidgets.QPushButton(self.widget)
        self.pushButton_lap.setMinimumSize(QtCore.QSize(150, 50))
        self.pushButton_lap.setObjectName("pushButton_lap")
        self.gridLayout_2.addWidget(self.pushButton_lap, 0, 0, 1, 1)
        self.pushButton_zhifangtu = QtWidgets.QPushButton(self.widget)
        self.pushButton_zhifangtu.setMinimumSize(QtCore.QSize(150, 50))
        self.pushButton_zhifangtu.setObjectName("pushButton_zhifangtu")
        self.gridLayout_2.addWidget(self.pushButton_zhifangtu, 0, 1, 1, 1)
        self.frame_2 = QtWidgets.QFrame(self.widget)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setGeometry(QtCore.QRect(10, 0, 51, 16))
        self.label.setMinimumSize(QtCore.QSize(40, 0))
        self.label.setObjectName("label")
        self.lineEdit_power_value = QtWidgets.QLineEdit(self.frame_2)
        self.lineEdit_power_value.setGeometry(QtCore.QRect(50, 20, 150, 19))
        self.lineEdit_power_value.setMinimumSize(QtCore.QSize(150, 0))
        self.lineEdit_power_value.setObjectName("lineEdit_power_value")
        self.gridLayout_2.addWidget(self.frame_2, 1, 0, 1, 1)
        self.pushButton_gama = QtWidgets.QPushButton(self.widget)
        self.pushButton_gama.setMinimumSize(QtCore.QSize(150, 50))
        self.pushButton_gama.setObjectName("pushButton_gama")
        self.gridLayout_2.addWidget(self.pushButton_gama, 1, 1, 1, 1)

        self.retranslateUi(Formwin3)
        QtCore.QMetaObject.connectSlotsByName(Formwin3)

    def retranslateUi(self, Formwin3):
        _translate = QtCore.QCoreApplication.translate
        Formwin3.setWindowTitle(_translate("Formwin3", "Form"))
        self.pushButton_load.setText(_translate("Formwin3", "选择图片"))
        self.label_jieguo.setText(_translate("Formwin3", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">结果</span></p></body></html>"))
        self.label_daichuli.setText(_translate("Formwin3", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">待处理</span></p></body></html>"))
        self.pushButton_save.setText(_translate("Formwin3", "保存图片"))
        self.pushButton_lap.setText(_translate("Formwin3", "拉普拉斯变化"))
        self.pushButton_zhifangtu.setText(_translate("Formwin3", "直方图均衡"))
        self.label.setText(_translate("Formwin3", "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">power:</span></p></body></html>"))
        self.pushButton_gama.setText(_translate("Formwin3", "伽马变化"))


class Ui_Formwin4(object):
    def setupUi(self, Formwin4):
        Formwin4.setObjectName("Formwin4")
        Formwin4.resize(1100, 800)
        Formwin4.setMinimumSize(QtCore.QSize(1100, 800))
        self.layoutWidget = QtWidgets.QWidget(Formwin4)
        self.layoutWidget.setGeometry(QtCore.QRect(110, 460, 751, 161))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 2, 1, 1)
        self.lineEdit_val_value = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_val_value.setMinimumSize(QtCore.QSize(150, 30))
        self.lineEdit_val_value.setObjectName("lineEdit_val_value")
        self.gridLayout.addWidget(self.lineEdit_val_value, 0, 3, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.pushButton_gmma = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_gmma.setMinimumSize(QtCore.QSize(100, 50))
        self.pushButton_gmma.setObjectName("pushButton_gmma")
        self.gridLayout.addWidget(self.pushButton_gmma, 0, 4, 1, 1)
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 3, 1, 1)
        self.lineEdit_mean_value = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_mean_value.setMinimumSize(QtCore.QSize(150, 30))
        self.lineEdit_mean_value.setObjectName("lineEdit_mean_value")
        self.gridLayout.addWidget(self.lineEdit_mean_value, 0, 1, 1, 1)
        self.lineEdit_n_value = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit_n_value.setMinimumSize(QtCore.QSize(150, 30))
        self.lineEdit_n_value.setObjectName("lineEdit_n_value")
        self.gridLayout.addWidget(self.lineEdit_n_value, 2, 1, 1, 1)
        self.pushButton_zhifangtu = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_zhifangtu.setMinimumSize(QtCore.QSize(100, 50))
        self.pushButton_zhifangtu.setObjectName("pushButton_zhifangtu")
        self.gridLayout.addWidget(self.pushButton_zhifangtu, 2, 4, 1, 1)
        self.widget = QtWidgets.QWidget(Formwin4)
        self.widget.setGeometry(QtCore.QRect(10, 50, 1041, 311))
        self.widget.setMinimumSize(QtCore.QSize(500, 0))
        self.widget.setObjectName("widget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_daichuli = QtWidgets.QLabel(self.widget)
        self.label_daichuli.setMinimumSize(QtCore.QSize(500, 0))
        self.label_daichuli.setObjectName("label_daichuli")
        self.horizontalLayout.addWidget(self.label_daichuli)
        self.label_jieguo = QtWidgets.QLabel(self.widget)
        self.label_jieguo.setMinimumSize(QtCore.QSize(500, 0))
        self.label_jieguo.setObjectName("label_jieguo")
        self.horizontalLayout.addWidget(self.label_jieguo)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 2)
        self.pushButton_load = QtWidgets.QPushButton(self.widget)
        self.pushButton_load.setMinimumSize(QtCore.QSize(500, 30))
        self.pushButton_load.setObjectName("pushButton_load")
        self.gridLayout_2.addWidget(self.pushButton_load, 1, 0, 1, 1)
        self.pushButton_save = QtWidgets.QPushButton(self.widget)
        self.pushButton_save.setMinimumSize(QtCore.QSize(500, 30))
        self.pushButton_save.setObjectName("pushButton_save")
        self.gridLayout_2.addWidget(self.pushButton_save, 1, 1, 1, 1)

        self.retranslateUi(Formwin4)
        QtCore.QMetaObject.connectSlotsByName(Formwin4)

    def retranslateUi(self, Formwin4):
        _translate = QtCore.QCoreApplication.translate
        Formwin4.setWindowTitle(_translate("Formwin4", "Form"))
        self.label_4.setText(_translate("Formwin4", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">val=</span></p></body></html>"))
        self.lineEdit_val_value.setText(_translate("Formwin4", "0.01"))
        self.label_3.setText(_translate("Formwin4", "<html><head/><body><p align=\"center\"><span style=\" font-size:11pt; font-weight:600;\">mean=</span></p></body></html>"))
        self.pushButton_gmma.setText(_translate("Formwin4", "添加高斯噪声"))
        self.label.setText(_translate("Formwin4", "<html><head/><body><p align=\"center\"><span style=\" font-size:11pt; font-weight:600;\">n=</span></p></body></html>"))
        self.label_2.setText(_translate("Formwin4", "(注：个数)"))
        self.lineEdit_mean_value.setText(_translate("Formwin4", "0"))
        self.lineEdit_n_value.setText(_translate("Formwin4", "1000"))
        self.pushButton_zhifangtu.setText(_translate("Formwin4", "添加椒盐噪声"))
        self.label_daichuli.setText(_translate("Formwin4", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">待处理</span></p></body></html>"))
        self.label_jieguo.setText(_translate("Formwin4", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">结果</span></p></body></html>"))
        self.pushButton_load.setText(_translate("Formwin4", "选择图片"))
        self.pushButton_save.setText(_translate("Formwin4", "保存图片"))


class Ui_Formwin5(object):
    def setupUi(self, Formwin5):
        Formwin5.setObjectName("Formwin5")
        Formwin5.resize(1100, 800)
        Formwin5.setMinimumSize(QtCore.QSize(1100, 800))
        self.layoutWidget = QtWidgets.QWidget(Formwin5)
        self.layoutWidget.setGeometry(QtCore.QRect(250, 40, 714, 622))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_daichuli = QtWidgets.QLabel(self.layoutWidget)
        self.label_daichuli.setMinimumSize(QtCore.QSize(500, 250))
        self.label_daichuli.setObjectName("label_daichuli")
        self.gridLayout.addWidget(self.label_daichuli, 0, 0, 1, 1)
        self.pushButton_load = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_load.setMinimumSize(QtCore.QSize(500, 50))
        self.pushButton_load.setObjectName("pushButton_load")
        self.gridLayout.addWidget(self.pushButton_load, 1, 0, 1, 1)
        self.label_jieguo = QtWidgets.QLabel(self.layoutWidget)
        self.label_jieguo.setMinimumSize(QtCore.QSize(500, 250))
        self.label_jieguo.setObjectName("label_jieguo")
        self.gridLayout.addWidget(self.label_jieguo, 2, 0, 1, 1)
        self.pushButton_save = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_save.setMinimumSize(QtCore.QSize(500, 50))
        self.pushButton_save.setObjectName("pushButton_save")
        self.gridLayout.addWidget(self.pushButton_save, 3, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)
        self.frame = QtWidgets.QFrame(self.layoutWidget)
        self.frame.setMinimumSize(QtCore.QSize(40, 0))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.layoutWidget1 = QtWidgets.QWidget(self.frame)
        self.layoutWidget1.setGeometry(QtCore.QRect(40, 0, 158, 580))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.formLayout = QtWidgets.QFormLayout(self.layoutWidget1)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.pushButton_zhongzhi = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_zhongzhi.setMinimumSize(QtCore.QSize(150, 60))
        self.pushButton_zhongzhi.setObjectName("pushButton_zhongzhi")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.pushButton_zhongzhi)
        self.pushButton_gauss = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_gauss.setMinimumSize(QtCore.QSize(150, 60))
        self.pushButton_gauss.setObjectName("pushButton_gauss")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.pushButton_gauss)
        self.pushButton_fangbo = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_fangbo.setMinimumSize(QtCore.QSize(150, 60))
        self.pushButton_fangbo.setObjectName("pushButton_fangbo")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.pushButton_fangbo)
        self.pushButton_junzhi = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_junzhi.setMinimumSize(QtCore.QSize(150, 60))
        self.pushButton_junzhi.setObjectName("pushButton_junzhi")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.pushButton_junzhi)
        self.pushButton_suangbian = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_suangbian.setMinimumSize(QtCore.QSize(150, 60))
        self.pushButton_suangbian.setObjectName("pushButton_suangbian")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.pushButton_suangbian)
        self.horizontalLayout.addWidget(self.frame)

        self.retranslateUi(Formwin5)
        QtCore.QMetaObject.connectSlotsByName(Formwin5)

    def retranslateUi(self, Formwin5):
        _translate = QtCore.QCoreApplication.translate
        Formwin5.setWindowTitle(_translate("Formwin5", "Form"))
        self.label_daichuli.setText(_translate("Formwin5", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">待处理</span></p></body></html>"))
        self.pushButton_load.setText(_translate("Formwin5", "选择图片"))
        self.label_jieguo.setText(_translate("Formwin5", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">结果</span></p></body></html>"))
        self.pushButton_save.setText(_translate("Formwin5", "保存图片"))
        self.pushButton_zhongzhi.setText(_translate("Formwin5", "中值滤波"))
        self.pushButton_gauss.setText(_translate("Formwin5", "高斯滤波"))
        self.pushButton_fangbo.setText(_translate("Formwin5", "方波滤波"))
        self.pushButton_junzhi.setText(_translate("Formwin5", "均值滤波"))
        self.pushButton_suangbian.setText(_translate("Formwin5", "双边滤波"))

class Ui_Formwin6(object):
    def setupUi(self, Formwin6):
        Formwin6.setObjectName("Formwin6")
        Formwin6.resize(1100, 800)
        Formwin6.setMinimumSize(QtCore.QSize(1100, 800))
        self.layoutWidget = QtWidgets.QWidget(Formwin6)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 90, 1008, 271))
        self.layoutWidget.setMinimumSize(QtCore.QSize(500, 0))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_jieguo = QtWidgets.QLabel(self.layoutWidget)
        self.label_jieguo.setMinimumSize(QtCore.QSize(500, 0))
        self.label_jieguo.setObjectName("label_jieguo")
        self.gridLayout.addWidget(self.label_jieguo, 0, 1, 1, 1)
        self.label_daichuli = QtWidgets.QLabel(self.layoutWidget)
        self.label_daichuli.setMinimumSize(QtCore.QSize(500, 0))
        self.label_daichuli.setObjectName("label_daichuli")
        self.gridLayout.addWidget(self.label_daichuli, 0, 0, 1, 1)
        self.pushButton_load = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_load.setMinimumSize(QtCore.QSize(500, 30))
        self.pushButton_load.setObjectName("pushButton_load")
        self.gridLayout.addWidget(self.pushButton_load, 1, 0, 1, 1)
        self.pushButton_save = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_save.setMinimumSize(QtCore.QSize(400, 30))
        self.pushButton_save.setObjectName("pushButton_save")
        self.gridLayout.addWidget(self.pushButton_save, 1, 1, 1, 1)
        self.layoutWidget1 = QtWidgets.QWidget(Formwin6)
        self.layoutWidget1.setGeometry(QtCore.QRect(270, 400, 511, 101))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_sift = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_sift.setMinimumSize(QtCore.QSize(80, 50))
        self.pushButton_sift.setObjectName("pushButton_sift")
        self.horizontalLayout.addWidget(self.pushButton_sift)
        self.pushButton_lunkuo = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_lunkuo.setMinimumSize(QtCore.QSize(80, 50))
        self.pushButton_lunkuo.setObjectName("pushButton_lunkuo")
        self.horizontalLayout.addWidget(self.pushButton_lunkuo)

        self.retranslateUi(Formwin6)
        QtCore.QMetaObject.connectSlotsByName(Formwin6)

    def retranslateUi(self, Formwin6):
        _translate = QtCore.QCoreApplication.translate
        Formwin6.setWindowTitle(_translate("Formwin6", "Form"))
        self.label_jieguo.setText(_translate("Formwin6", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">结果</span></p></body></html>"))
        self.label_daichuli.setText(_translate("Formwin6", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">待处理</span></p></body></html>"))
        self.pushButton_load.setText(_translate("Formwin6", "选择图片"))
        self.pushButton_save.setText(_translate("Formwin6", "保存图片"))
        self.pushButton_sift.setText(_translate("Formwin6", "sift检测"))
        self.pushButton_lunkuo.setText(_translate("Formwin6", "轮廓检测"))



class Ui_Formwin7(object):
    def setupUi(self, Formwin7):
        Formwin7.setObjectName("Formwin7")
        Formwin7.resize(1100, 800)
        Formwin7.setMinimumSize(QtCore.QSize(1100, 800))
        self.layoutWidget = QtWidgets.QWidget(Formwin7)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 50, 1008, 311))
        self.layoutWidget.setMinimumSize(QtCore.QSize(500, 0))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_load = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_load.setMinimumSize(QtCore.QSize(400, 50))
        self.pushButton_load.setObjectName("pushButton_load")
        self.gridLayout.addWidget(self.pushButton_load, 1, 0, 1, 1)
        self.label_jieguo = QtWidgets.QLabel(self.layoutWidget)
        self.label_jieguo.setMinimumSize(QtCore.QSize(500, 0))
        self.label_jieguo.setObjectName("label_jieguo")
        self.gridLayout.addWidget(self.label_jieguo, 0, 1, 1, 1)
        self.label_daichuli = QtWidgets.QLabel(self.layoutWidget)
        self.label_daichuli.setMinimumSize(QtCore.QSize(500, 0))
        self.label_daichuli.setObjectName("label_daichuli")
        self.gridLayout.addWidget(self.label_daichuli, 0, 0, 1, 1)
        self.pushButton_save = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_save.setMinimumSize(QtCore.QSize(400, 50))
        self.pushButton_save.setObjectName("pushButton_save")
        self.gridLayout.addWidget(self.pushButton_save, 1, 1, 1, 1)
        self.layoutWidget1 = QtWidgets.QWidget(Formwin7)
        self.layoutWidget1.setGeometry(QtCore.QRect(180, 390, 661, 111))
        self.layoutWidget1.setMinimumSize(QtCore.QSize(80, 50))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_xiufu = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_xiufu.setMinimumSize(QtCore.QSize(80, 50))
        self.pushButton_xiufu.setObjectName("pushButton_xiufu")
        self.horizontalLayout.addWidget(self.pushButton_xiufu)
        self.pushButton_SRGAN = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_SRGAN.setMinimumSize(QtCore.QSize(80, 50))
        self.pushButton_SRGAN.setObjectName("pushButton_SRGAN")
        self.horizontalLayout.addWidget(self.pushButton_SRGAN)
        self.pushButton_MPRnet = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_MPRnet.setMinimumSize(QtCore.QSize(80, 50))
        self.pushButton_MPRnet.setObjectName("pushButton_MPRnet")
        self.horizontalLayout.addWidget(self.pushButton_MPRnet)
        self.pushButton_DeblurGan = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_DeblurGan.setMinimumSize(QtCore.QSize(80, 50))
        self.pushButton_DeblurGan.setObjectName("pushButton_DeblurGan")
        self.horizontalLayout.addWidget(self.pushButton_DeblurGan)
        self.pushButton_Deblurganv2 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_Deblurganv2.setMinimumSize(QtCore.QSize(80, 50))
        self.pushButton_Deblurganv2.setObjectName("pushButton_Deblurganv2")
        self.horizontalLayout.addWidget(self.pushButton_Deblurganv2)

        self.retranslateUi(Formwin7)
        QtCore.QMetaObject.connectSlotsByName(Formwin7)

    def retranslateUi(self, Formwin7):
        _translate = QtCore.QCoreApplication.translate
        Formwin7.setWindowTitle(_translate("Formwin7", "Form"))
        self.pushButton_load.setText(_translate("Formwin7", "选择图片"))
        self.label_jieguo.setText(_translate("Formwin7", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">结果</span></p></body></html>"))
        self.label_daichuli.setText(_translate("Formwin7", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">待处理</span></p></body></html>"))
        self.pushButton_save.setText(_translate("Formwin7", "保存图片"))
        self.pushButton_xiufu.setText(_translate("Formwin7", "图像修复"))
        self.pushButton_SRGAN.setText(_translate("Formwin7", "SRGAN"))
        self.pushButton_MPRnet.setText(_translate("Formwin7", "MPRNet"))
        self.pushButton_DeblurGan.setText(_translate("Formwin7", "DeblurGan"))
        self.pushButton_Deblurganv2.setText(_translate("Formwin7", "DeblurGan-v2"))