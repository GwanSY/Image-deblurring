# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Fromwin2.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Formwin2(object):
    def setupUi(self, Formwin2):
        Formwin2.setObjectName("Formwin2")
        Formwin2.resize(571, 388)
        self.pushButton_load = QtWidgets.QPushButton(Formwin2)
        self.pushButton_load.setGeometry(QtCore.QRect(110, 230, 101, 31))
        self.pushButton_load.setObjectName("pushButton_load")
        self.pushButton_save = QtWidgets.QPushButton(Formwin2)
        self.pushButton_save.setGeometry(QtCore.QRect(350, 230, 101, 31))
        self.pushButton_save.setObjectName("pushButton_save")
        self.label_tips = QtWidgets.QLabel(Formwin2)
        self.label_tips.setGeometry(QtCore.QRect(30, 290, 51, 41))
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
        self.splitter = QtWidgets.QSplitter(Formwin2)
        self.splitter.setGeometry(QtCore.QRect(40, 30, 501, 171))
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.label_daichuli = QtWidgets.QLabel(self.splitter)
        self.label_daichuli.setObjectName("label_daichuli")
        self.label_jieguo = QtWidgets.QLabel(self.splitter)
        self.label_jieguo.setObjectName("label_jieguo")

        self.retranslateUi(Formwin2)
        QtCore.QMetaObject.connectSlotsByName(Formwin2)

    def retranslateUi(self, Formwin2):
        _translate = QtCore.QCoreApplication.translate
        Formwin2.setWindowTitle(_translate("Formwin2", "Form"))
        self.pushButton_load.setText(_translate("Formwin2", "选择图片"))
        self.pushButton_save.setText(_translate("Formwin2", "保存图片"))
        self.label_tips.setText(_translate("Formwin2", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">操作：</span></p></body></html>"))
        self.pushButton_fun0.setText(_translate("Formwin2", "水平翻转"))
        self.pushButton_fun1.setText(_translate("Formwin2", "垂直翻转"))
        self.pushButton_fun11.setText(_translate("Formwin2", "沿xy轴翻转"))
        self.label_daichuli.setText(_translate("Formwin2", "<html><head/><body><p align=\"center\"><span style=\" font-size:11pt; font-weight:600;\">待处理</span></p></body></html>"))
        self.label_jieguo.setText(_translate("Formwin2", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">结果</span></p></body></html>"))
