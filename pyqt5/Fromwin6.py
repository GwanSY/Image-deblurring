# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Fromwin6.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Formwin6(object):
    def setupUi(self, Formwin6):
        Formwin6.setObjectName("Formwin6")
        Formwin6.resize(1100, 700)
        Formwin6.setMinimumSize(QtCore.QSize(1100, 700))
        Formwin6.setMaximumSize(QtCore.QSize(1100, 700))
        self.layoutWidget = QtWidgets.QWidget(Formwin6)
        self.layoutWidget.setGeometry(QtCore.QRect(40, 90, 1008, 271))
        self.layoutWidget.setMinimumSize(QtCore.QSize(500, 0))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_jieguo = QtWidgets.QLabel(self.layoutWidget)
        self.label_jieguo.setMinimumSize(QtCore.QSize(500, 0))
        self.label_jieguo.setStyleSheet("")
        self.label_jieguo.setObjectName("label_jieguo")
        self.gridLayout.addWidget(self.label_jieguo, 0, 1, 1, 1)
        self.label_daichuli = QtWidgets.QLabel(self.layoutWidget)
        self.label_daichuli.setMinimumSize(QtCore.QSize(500, 0))
        self.label_daichuli.setStyleSheet("")
        self.label_daichuli.setObjectName("label_daichuli")
        self.gridLayout.addWidget(self.label_daichuli, 0, 0, 1, 1)
        self.pushButton_load = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_load.setMinimumSize(QtCore.QSize(500, 50))
        self.pushButton_load.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.pushButton_load.setObjectName("pushButton_load")
        self.gridLayout.addWidget(self.pushButton_load, 1, 0, 1, 1)
        self.pushButton_save = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_save.setMinimumSize(QtCore.QSize(400, 50))
        self.pushButton_save.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.pushButton_save.setObjectName("pushButton_save")
        self.gridLayout.addWidget(self.pushButton_save, 1, 1, 1, 1)
        self.layoutWidget1 = QtWidgets.QWidget(Formwin6)
        self.layoutWidget1.setGeometry(QtCore.QRect(290, 400, 511, 101))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_sift = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_sift.setMinimumSize(QtCore.QSize(80, 50))
        self.pushButton_sift.setStyleSheet("border-radius:30px;\n"
"background-color: rgb(255, 255, 255);")
        self.pushButton_sift.setObjectName("pushButton_sift")
        self.horizontalLayout.addWidget(self.pushButton_sift)
        self.pushButton_lunkuo = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_lunkuo.setMinimumSize(QtCore.QSize(80, 50))
        self.pushButton_lunkuo.setStyleSheet("border-radius:30px;background-color: rgb(255, 255, 255);")
        self.pushButton_lunkuo.setObjectName("pushButton_lunkuo")
        self.horizontalLayout.addWidget(self.pushButton_lunkuo)

        self.retranslateUi(Formwin6)
        QtCore.QMetaObject.connectSlotsByName(Formwin6)

    def retranslateUi(self, Formwin6):
        _translate = QtCore.QCoreApplication.translate
        Formwin6.setWindowTitle(_translate("Formwin6", "Form"))
        self.label_jieguo.setText(_translate("Formwin6", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600; color:#ffffff;\">结果</span></p></body></html>"))
        self.label_daichuli.setText(_translate("Formwin6", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600; color:#ffffff;\">待处理</span></p></body></html>"))
        self.pushButton_load.setText(_translate("Formwin6", "选择图片"))
        self.pushButton_save.setText(_translate("Formwin6", "保存图片"))
        self.pushButton_sift.setText(_translate("Formwin6", "sift检测"))
        self.pushButton_lunkuo.setText(_translate("Formwin6", "轮廓检测"))
