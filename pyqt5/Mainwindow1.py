# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Mainwindow1.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Mainwindow(object):
    def setupUi(self, Mainwindow):
        Mainwindow.setObjectName("Mainwindow")
        Mainwindow.resize(1800, 800)
        Mainwindow.setMinimumSize(QtCore.QSize(1500, 800))
        self.frame = QtWidgets.QFrame(Mainwindow)
        self.frame.setGeometry(QtCore.QRect(290, 30, 1501, 700))
        self.frame.setMinimumSize(QtCore.QSize(1150, 700))
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
        Formwin1.resize(1600, 900)
        self.label_daichuli = QtWidgets.QLabel(Formwin1)
        self.label_daichuli.setGeometry(QtCore.QRect(80, 50, 151, 151))
        self.label_daichuli.setObjectName("label_daichuli")
        self.label_jieguo = QtWidgets.QLabel(Formwin1)
        self.label_jieguo.setGeometry(QtCore.QRect(340, 50, 151, 151))
        self.label_jieguo.setObjectName("label_jieguo")
        self.pushButton_load = QtWidgets.QPushButton(Formwin1)
        self.pushButton_load.setGeometry(QtCore.QRect(100, 220, 101, 31))
        self.pushButton_load.setObjectName("pushButton_load")
        self.pushButton_save = QtWidgets.QPushButton(Formwin1)
        self.pushButton_save.setGeometry(QtCore.QRect(350, 220, 101, 31))
        self.pushButton_save.setObjectName("pushButton_save")
        self.label_tips = QtWidgets.QLabel(Formwin1)
        self.label_tips.setGeometry(QtCore.QRect(30, 280, 51, 41))
        self.label_tips.setObjectName("label_tips")
        self.pushButton_turntoGray = QtWidgets.QPushButton(Formwin1)
        self.pushButton_turntoGray.setGeometry(QtCore.QRect(120, 290, 101, 31))
        self.pushButton_turntoGray.setObjectName("pushButton_turntoGray")
        self.pushButton_turntotwo = QtWidgets.QPushButton(Formwin1)
        self.pushButton_turntotwo.setGeometry(QtCore.QRect(270, 290, 101, 31))
        self.pushButton_turntotwo.setObjectName("pushButton_turntotwo")

        self.retranslateUi(Formwin1)
        QtCore.QMetaObject.connectSlotsByName(Formwin1)

    def retranslateUi(self, Formwin1):
        _translate = QtCore.QCoreApplication.translate
        Formwin1.setWindowTitle(_translate("Formwin1", "Form"))
        self.label_daichuli.setText(_translate("Formwin1", "待处理"))
        self.label_jieguo.setText(_translate("Formwin1", "结果"))
        self.pushButton_load.setText(_translate("Formwin1", "选择图片"))
        self.pushButton_save.setText(_translate("Formwin1", "保存图片"))
        self.label_tips.setText(_translate("Formwin1", "操作："))
        self.pushButton_turntoGray.setText(_translate("Formwin1", "转成灰度图"))
        self.pushButton_turntotwo.setText(_translate("Formwin1", "将图片二值化"))


class Ui_Formwin2(object):
    def setupUi(self, Formwin2):
        Formwin2.setObjectName("Formwin2")
        Formwin2.resize(571, 388)
        self.label_daichuli = QtWidgets.QLabel(Formwin2)
        self.label_daichuli.setGeometry(QtCore.QRect(80, 50, 151, 151))
        self.label_daichuli.setObjectName("label_daichuli")
        self.label_jieguo = QtWidgets.QLabel(Formwin2)
        self.label_jieguo.setGeometry(QtCore.QRect(340, 50, 151, 151))
        self.label_jieguo.setObjectName("label_jieguo")
        self.pushButton_load = QtWidgets.QPushButton(Formwin2)
        self.pushButton_load.setGeometry(QtCore.QRect(100, 220, 101, 31))
        self.pushButton_load.setObjectName("pushButton_load")
        self.pushButton_save = QtWidgets.QPushButton(Formwin2)
        self.pushButton_save.setGeometry(QtCore.QRect(350, 220, 101, 31))
        self.pushButton_save.setObjectName("pushButton_save")
        self.label_tips = QtWidgets.QLabel(Formwin2)
        self.label_tips.setGeometry(QtCore.QRect(30, 280, 51, 41))
        self.label_tips.setObjectName("label_tips")
        self.pushButton_fun0 = QtWidgets.QPushButton(Formwin2)
        self.pushButton_fun0.setGeometry(QtCore.QRect(110, 290, 101, 31))
        self.pushButton_fun0.setObjectName("pushButton_fun0")
        self.pushButton_fun1 = QtWidgets.QPushButton(Formwin2)
        self.pushButton_fun1.setGeometry(QtCore.QRect(230, 290, 101, 31))
        self.pushButton_fun1.setObjectName("pushButton_fun1")
        self.pushButton_fun11 = QtWidgets.QPushButton(Formwin2)
        self.pushButton_fun11.setGeometry(QtCore.QRect(350, 290, 101, 31))
        self.pushButton_fun11.setObjectName("pushButton_fun11")

        self.retranslateUi(Formwin2)
        QtCore.QMetaObject.connectSlotsByName(Formwin2)

    def retranslateUi(self, Formwin2):
        _translate = QtCore.QCoreApplication.translate
        Formwin2.setWindowTitle(_translate("Formwin2", "Form"))
        self.label_daichuli.setText(_translate("Formwin2", "待处理"))
        self.label_jieguo.setText(_translate("Formwin2", "结果"))
        self.pushButton_load.setText(_translate("Formwin2", "选择图片"))
        self.pushButton_save.setText(_translate("Formwin2", "保存图片"))
        self.label_tips.setText(_translate("Formwin2", "操作："))
        self.pushButton_fun0.setText(_translate("Formwin2", "水平翻转"))
        self.pushButton_fun1.setText(_translate("Formwin2", "垂直翻转"))
        self.pushButton_fun11.setText(_translate("Formwin2", "沿xy轴翻转"))


class Ui_Formwin3(object):
    def setupUi(self, Formwin3):
        Formwin3.setObjectName("Formwin3")
        Formwin3.resize(571, 388)
        self.label_daichuli = QtWidgets.QLabel(Formwin3)
        self.label_daichuli.setGeometry(QtCore.QRect(80, 50, 151, 151))
        self.label_daichuli.setObjectName("label_daichuli")
        self.label_jieguo = QtWidgets.QLabel(Formwin3)
        self.label_jieguo.setGeometry(QtCore.QRect(340, 50, 151, 151))
        self.label_jieguo.setObjectName("label_jieguo")
        self.pushButton_load = QtWidgets.QPushButton(Formwin3)
        self.pushButton_load.setGeometry(QtCore.QRect(100, 220, 101, 31))
        self.pushButton_load.setObjectName("pushButton_load")
        self.pushButton_save = QtWidgets.QPushButton(Formwin3)
        self.pushButton_save.setGeometry(QtCore.QRect(350, 220, 101, 31))
        self.pushButton_save.setObjectName("pushButton_save")
        self.label_tips = QtWidgets.QLabel(Formwin3)
        self.label_tips.setGeometry(QtCore.QRect(40, 280, 51, 41))
        self.label_tips.setObjectName("label_tips")
        self.pushButton_lap = QtWidgets.QPushButton(Formwin3)
        self.pushButton_lap.setGeometry(QtCore.QRect(160, 290, 101, 31))
        self.pushButton_lap.setObjectName("pushButton_lap")
        self.pushButton_zhifangtu = QtWidgets.QPushButton(Formwin3)
        self.pushButton_zhifangtu.setGeometry(QtCore.QRect(290, 290, 101, 31))
        self.pushButton_zhifangtu.setObjectName("pushButton_zhifangtu")
        self.pushButton_gama = QtWidgets.QPushButton(Formwin3)
        self.pushButton_gama.setGeometry(QtCore.QRect(290, 340, 101, 31))
        self.pushButton_gama.setObjectName("pushButton_gama")
        self.label = QtWidgets.QLabel(Formwin3)
        self.label.setGeometry(QtCore.QRect(80, 340, 51, 31))
        self.label.setObjectName("label")
        self.lineEdit_power_value = QtWidgets.QLineEdit(Formwin3)
        self.lineEdit_power_value.setGeometry(QtCore.QRect(150, 340, 111, 31))
        self.lineEdit_power_value.setObjectName("lineEdit_power_value")

        self.retranslateUi(Formwin3)
        QtCore.QMetaObject.connectSlotsByName(Formwin3)

    def retranslateUi(self, Formwin3):
        _translate = QtCore.QCoreApplication.translate
        Formwin3.setWindowTitle(_translate("Formwin3", "Form"))
        self.label_daichuli.setText(_translate("Formwin3", "待处理"))
        self.label_jieguo.setText(_translate("Formwin3", "结果"))
        self.pushButton_load.setText(_translate("Formwin3", "选择图片"))
        self.pushButton_save.setText(_translate("Formwin3", "保存图片"))
        self.label_tips.setText(_translate("Formwin3", "操作："))
        self.pushButton_lap.setText(_translate("Formwin3", "拉普拉斯变化"))
        self.pushButton_zhifangtu.setText(_translate("Formwin3", "直方图均衡"))
        self.pushButton_gama.setText(_translate("Formwin3", "伽马变化"))
        self.label.setText(_translate("Formwin3", "power1="))
        self.lineEdit_power_value.setText(_translate("Formwin3", "1.5"))


class Ui_Formwin4(object):
    def setupUi(self, Formwin4):
        Formwin4.setObjectName("Formwin4")
        Formwin4.resize(571, 388)
        self.label_daichuli = QtWidgets.QLabel(Formwin4)
        self.label_daichuli.setGeometry(QtCore.QRect(80, 50, 151, 151))
        self.label_daichuli.setObjectName("label_daichuli")
        self.label_jieguo = QtWidgets.QLabel(Formwin4)
        self.label_jieguo.setGeometry(QtCore.QRect(340, 50, 151, 151))
        self.label_jieguo.setObjectName("label_jieguo")
        self.pushButton_load = QtWidgets.QPushButton(Formwin4)
        self.pushButton_load.setGeometry(QtCore.QRect(100, 220, 101, 31))
        self.pushButton_load.setObjectName("pushButton_load")
        self.pushButton_save = QtWidgets.QPushButton(Formwin4)
        self.pushButton_save.setGeometry(QtCore.QRect(350, 220, 101, 31))
        self.pushButton_save.setObjectName("pushButton_save")
        self.label_tips = QtWidgets.QLabel(Formwin4)
        self.label_tips.setGeometry(QtCore.QRect(10, 250, 41, 41))
        self.label_tips.setObjectName("label_tips")
        self.pushButton_zhifangtu = QtWidgets.QPushButton(Formwin4)
        self.pushButton_zhifangtu.setGeometry(QtCore.QRect(340, 280, 101, 31))
        self.pushButton_zhifangtu.setObjectName("pushButton_zhifangtu")
        self.pushButton_gama = QtWidgets.QPushButton(Formwin4)
        self.pushButton_gama.setGeometry(QtCore.QRect(340, 330, 101, 31))
        self.pushButton_gama.setObjectName("pushButton_gama")
        self.label = QtWidgets.QLabel(Formwin4)
        self.label.setGeometry(QtCore.QRect(160, 330, 21, 31))
        self.label.setObjectName("label")
        self.lineEdit_n_value = QtWidgets.QLineEdit(Formwin4)
        self.lineEdit_n_value.setGeometry(QtCore.QRect(180, 330, 101, 31))
        self.lineEdit_n_value.setObjectName("lineEdit_n_value")
        self.label_2 = QtWidgets.QLabel(Formwin4)
        self.label_2.setGeometry(QtCore.QRect(200, 360, 71, 21))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Formwin4)
        self.label_3.setGeometry(QtCore.QRect(100, 280, 31, 31))
        self.label_3.setObjectName("label_3")
        self.lineEdit_mean_value = QtWidgets.QLineEdit(Formwin4)
        self.lineEdit_mean_value.setGeometry(QtCore.QRect(140, 280, 41, 31))
        self.lineEdit_mean_value.setObjectName("lineEdit_mean_value")
        self.lineEdit_val_value = QtWidgets.QLineEdit(Formwin4)
        self.lineEdit_val_value.setGeometry(QtCore.QRect(250, 280, 41, 31))
        self.lineEdit_val_value.setObjectName("lineEdit_val_value")
        self.label_4 = QtWidgets.QLabel(Formwin4)
        self.label_4.setGeometry(QtCore.QRect(220, 280, 31, 31))
        self.label_4.setObjectName("label_4")

        self.retranslateUi(Formwin4)
        QtCore.QMetaObject.connectSlotsByName(Formwin4)

    def retranslateUi(self, Formwin4):
        _translate = QtCore.QCoreApplication.translate
        Formwin4.setWindowTitle(_translate("Formwin4", "Form"))
        self.label_daichuli.setText(_translate("Formwin4", "待处理"))
        self.label_jieguo.setText(_translate("Formwin4", "结果"))
        self.pushButton_load.setText(_translate("Formwin4", "选择图片"))
        self.pushButton_save.setText(_translate("Formwin4", "保存图片"))
        self.label_tips.setText(_translate("Formwin4", "操作："))
        self.pushButton_zhifangtu.setText(_translate("Formwin4", "添加高斯噪声"))
        self.pushButton_gama.setText(_translate("Formwin4", "添加椒盐噪声"))
        self.label.setText(_translate("Formwin4", "n="))
        self.lineEdit_n_value.setText(_translate("Formwin4", "1000"))
        self.label_2.setText(_translate("Formwin4", "(注：个数)"))
        self.label_3.setText(_translate("Formwin4", "mean="))
        self.lineEdit_mean_value.setText(_translate("Formwin4", "0"))
        self.lineEdit_val_value.setText(_translate("Formwin4", "0.01"))
        self.label_4.setText(_translate("Formwin4", "val="))


class Ui_Formwin5(object):
    def setupUi(self, Formwin5):
        Formwin5.setObjectName("Formwin5")
        Formwin5.resize(571, 388)
        self.label_daichuli = QtWidgets.QLabel(Formwin5)
        self.label_daichuli.setGeometry(QtCore.QRect(80, 50, 151, 151))
        self.label_daichuli.setObjectName("label_daichuli")
        self.label_jieguo = QtWidgets.QLabel(Formwin5)
        self.label_jieguo.setGeometry(QtCore.QRect(340, 50, 151, 151))
        self.label_jieguo.setObjectName("label_jieguo")
        self.pushButton_load = QtWidgets.QPushButton(Formwin5)
        self.pushButton_load.setGeometry(QtCore.QRect(100, 220, 101, 31))
        self.pushButton_load.setObjectName("pushButton_load")
        self.pushButton_save = QtWidgets.QPushButton(Formwin5)
        self.pushButton_save.setGeometry(QtCore.QRect(350, 220, 101, 31))
        self.pushButton_save.setObjectName("pushButton_save")
        self.pushButton_fangbo = QtWidgets.QPushButton(Formwin5)
        self.pushButton_fangbo.setGeometry(QtCore.QRect(100, 280, 101, 31))
        self.pushButton_fangbo.setObjectName("pushButton_fangbo")
        self.pushButton_zhongzhi = QtWidgets.QPushButton(Formwin5)
        self.pushButton_zhongzhi.setGeometry(QtCore.QRect(230, 280, 101, 31))
        self.pushButton_zhongzhi.setObjectName("pushButton_zhongzhi")
        self.pushButton_junzhi = QtWidgets.QPushButton(Formwin5)
        self.pushButton_junzhi.setGeometry(QtCore.QRect(230, 330, 101, 31))
        self.pushButton_junzhi.setObjectName("pushButton_junzhi")
        self.pushButton_gauss = QtWidgets.QPushButton(Formwin5)
        self.pushButton_gauss.setGeometry(QtCore.QRect(100, 330, 101, 31))
        self.pushButton_gauss.setObjectName("pushButton_gauss")
        self.pushButton_suangbian = QtWidgets.QPushButton(Formwin5)
        self.pushButton_suangbian.setGeometry(QtCore.QRect(360, 280, 101, 31))
        self.pushButton_suangbian.setObjectName("pushButton_suangbian")

        self.retranslateUi(Formwin5)
        QtCore.QMetaObject.connectSlotsByName(Formwin5)

    def retranslateUi(self, Formwin5):
        _translate = QtCore.QCoreApplication.translate
        Formwin5.setWindowTitle(_translate("Formwin5", "Form"))
        self.label_daichuli.setText(_translate("Formwin5", "待处理"))
        self.label_jieguo.setText(_translate("Formwin5", "结果"))
        self.pushButton_load.setText(_translate("Formwin5", "选择图片"))
        self.pushButton_save.setText(_translate("Formwin5", "保存图片"))
        self.pushButton_fangbo.setText(_translate("Formwin5", "方波滤波"))
        self.pushButton_zhongzhi.setText(_translate("Formwin5", "中值滤波"))
        self.pushButton_junzhi.setText(_translate("Formwin5", "均值滤波"))
        self.pushButton_gauss.setText(_translate("Formwin5", "高斯滤波"))
        self.pushButton_suangbian.setText(_translate("Formwin5", "双边滤波"))


class Ui_Formwin6(object):
    def setupUi(self, Formwin6):
        Formwin6.setObjectName("Formwin6")
        Formwin6.resize(571, 388)
        self.label_daichuli = QtWidgets.QLabel(Formwin6)
        self.label_daichuli.setGeometry(QtCore.QRect(70, 40, 201, 201))
        self.label_daichuli.setObjectName("label_daichuli")
        self.label_jieguo = QtWidgets.QLabel(Formwin6)
        self.label_jieguo.setGeometry(QtCore.QRect(300, 40, 201, 201))
        self.label_jieguo.setObjectName("label_jieguo")
        self.pushButton_load = QtWidgets.QPushButton(Formwin6)
        self.pushButton_load.setGeometry(QtCore.QRect(100, 260, 101, 31))
        self.pushButton_load.setObjectName("pushButton_load")
        self.pushButton_save = QtWidgets.QPushButton(Formwin6)
        self.pushButton_save.setGeometry(QtCore.QRect(350, 260, 101, 31))
        self.pushButton_save.setObjectName("pushButton_save")
        self.pushButton_sift = QtWidgets.QPushButton(Formwin6)
        self.pushButton_sift.setGeometry(QtCore.QRect(100, 320, 101, 31))
        self.pushButton_sift.setObjectName("pushButton_sift")
        self.pushButton_lunkuo = QtWidgets.QPushButton(Formwin6)
        self.pushButton_lunkuo.setGeometry(QtCore.QRect(230, 320, 101, 31))
        self.pushButton_lunkuo.setObjectName("pushButton_lunkuo")

        self.retranslateUi(Formwin6)
        QtCore.QMetaObject.connectSlotsByName(Formwin6)

    def retranslateUi(self, Formwin6):
        _translate = QtCore.QCoreApplication.translate
        Formwin6.setWindowTitle(_translate("Formwin6", "Form"))
        self.label_daichuli.setText(_translate("Formwin6", "待处理"))
        self.label_jieguo.setText(_translate("Formwin6", "结果"))
        self.pushButton_load.setText(_translate("Formwin6", "选择图片"))
        self.pushButton_save.setText(_translate("Formwin6", "保存图片"))
        self.pushButton_sift.setText(_translate("Formwin6", "sift检测"))
        self.pushButton_lunkuo.setText(_translate("Formwin6", "轮廓检测"))


class Ui_Formwin7(object):
    def setupUi(self, Formwin7):
        Formwin7.setObjectName("Formwin7")
        Formwin7.resize(571, 388)
        self.label_daichuli = QtWidgets.QLabel(Formwin7)
        self.label_daichuli.setGeometry(QtCore.QRect(70, 40, 201, 201))
        self.label_daichuli.setObjectName("label_daichuli")
        self.label_jieguo = QtWidgets.QLabel(Formwin7)
        self.label_jieguo.setGeometry(QtCore.QRect(300, 40, 201, 201))
        self.label_jieguo.setObjectName("label_jieguo")
        self.pushButton_load = QtWidgets.QPushButton(Formwin7)
        self.pushButton_load.setGeometry(QtCore.QRect(100, 260, 101, 31))
        self.pushButton_load.setObjectName("pushButton_load")
        self.pushButton_save = QtWidgets.QPushButton(Formwin7)
        self.pushButton_save.setGeometry(QtCore.QRect(350, 260, 101, 31))
        self.pushButton_save.setObjectName("pushButton_save")
        self.pushButton_xiufu = QtWidgets.QPushButton(Formwin7)
        self.pushButton_xiufu.setGeometry(QtCore.QRect(230, 320, 101, 31))
        self.pushButton_xiufu.setObjectName("pushButton_xiufu")

        self.retranslateUi(Formwin7)
        QtCore.QMetaObject.connectSlotsByName(Formwin7)

    def retranslateUi(self, Formwin7):
        _translate = QtCore.QCoreApplication.translate
        Formwin7.setWindowTitle(_translate("Formwin7", "Form"))
        self.label_daichuli.setText(_translate("Formwin7", "待处理"))
        self.label_jieguo.setText(_translate("Formwin7", "结果"))
        self.pushButton_load.setText(_translate("Formwin7", "选择图片"))
        self.pushButton_save.setText(_translate("Formwin7", "保存图片"))
        self.pushButton_xiufu.setText(_translate("Formwin7", "图像修复"))