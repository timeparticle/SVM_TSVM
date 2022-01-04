import random

from numpy import *
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup=('lin',0)):
        """
        Args:
            dataMatIn    数据集
            classLabels  类别标签
            C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
                控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
                可以通过调节该参数达到不同的结果。
            toler   容错率
            kTup    包含核函数信息的元组
        """
        self.X = mat(dataMatIn)   # matrix m x n
        # 数据的行数 列数
        self.m, self.n = shape(dataMatIn)
        self.Ytrain = reshape(classLabels,(self.m,1))   #ndarray m x 1
        self.C = C
        self.tol = toler
        self.alphas = zeros((self.m,1))
        self.ws = zeros((self.n, 1))
        self.b = 0
        self.Ytrain_pred=zeros((self.m,1))
        self.kTup=kTup
        # 误差缓存，第一列给出的是eCache是否有效的标志位，第二列给出的是实际的E值。
        self.eCache = mat(zeros((self.m, 2)))
        # 计算m行m列的 k(xi,xj)矩阵
        self.K = mat(zeros((self.m, self.m)))
        self.kernelMatrix()

    def calcEk(self, k):
        fXk = float(multiply(self.Ytrain , self.alphas).T * (self.K[:, k]) + self.b)
        Ek = fXk - float(self.Ytrain[k])
        return Ek

    def selectJrand(self, i):
        j = i
        while j == i:
            j = int(random.uniform(0, self.m))
        return j

    def selectJ(self,i, Ei):
        #启发式选择alpha2
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        # 首先将输入值Ei在缓存中设置成为有效的。这里的有效意味着它已经计算好了。
        self.eCache[i] = [1, Ei]
        # 非零E值的行的list列表，所对应的alpha值
        validEcacheList = nonzero(self.eCache[:, 0].A)[0]
        if (len(validEcacheList)) > 1:
            for k in validEcacheList:  # 在所有的值上进行循环，并选择其中使得改变最大的那个值
                if k == i:
                    continue  # don't calc for i, waste of time
                Ek = self.calcEk(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    # 选择具有最大步长的j
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:  # 如果是第一次循环，则随机选择一个alpha值
            j = self.selectJrand(i)
            Ej = self.calcEk(j)
        return j, Ej

    def updateEk(self, k):
        # 求 误差：预测值-真实值的差
        Ek = self.calcEk(k)
        self.eCache[k] = [1, Ek]

    def clipAlpha(self,aj, H, L):
        #clipAlpha(调整aj的值，使aj处于 L<=aj<=H)
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def innerL(self,i):
        """
        内循环代码
        Args:i   具体的某一行   oS  optStruct对象
        Returns: 0 找不到最优的值   1   找到了最优的值，并且oS.Cache到缓存中
        """
        # Ek=fi-Yi
        Ei = self.calcEk(i)
        # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
        # 检验训练样本(xi, yi)是否满足KKT条件
        # yi*f(i) >= 1 and alpha = 0 (outside the boundary)
        # yi*f(i) == 1 and 0<alpha< C (on the boundary)
        # yi*f(i) <= 1 and alpha = C (between the boundary)
        # Yi(fi-Yi)= Yi*fi-1   |Yi*fi-1|>tol
        if ((self.Ytrain[i] * Ei < -self.tol) and (self.alphas[i] < self.C)) or ((self.Ytrain[i] * Ei > self.tol) and (self.alphas[i] > 0)):
            # 选择最大的误差对应的j进行优化。效果更明显
            j, Ej = self.selectJ(i, Ei)
            alphaIold = self.alphas[i].copy()
            alphaJold = self.alphas[j].copy()
            # 寻找0-C之间的alpha。如果L==H return 0  , P144 7.104
            if self.Ytrain[i] != self.Ytrain[j]:
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L == H:
                # print("L==H")
                return 0
            # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
            # 参考《统计学习方法》李航-P145 7.107<序列最小最优化算法>
            eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]  # eta = -(k11+k22-2K12) = -(K1+K2)^2 <=0
            if eta >= 0:
                print("eta>=0")
                return 0

            self.alphas[j] -= self.Ytrain[j] * (Ei - Ej) / eta       # 未剪辑的a2=alphas[j] 《统计学习方法》7.106
            self.alphas[j] = self.clipAlpha(self.alphas[j], H, L)          # 剪辑后的a2  《统计学习方法》7.108
            self.updateEk(j)                                      # 更新误差缓存 E2
            if abs(self.alphas[j] - alphaJold) < 0.00001:         # 检查a2 是否只是轻微的改变，如果是的话，就退出for循环。
                # print("j not moving enough")
                return 0
            # alpha[i] = a1_new =a1_old+y1*y2(a2_old-a2_new)
            self.alphas[i] += self.Ytrain[j] * self.Ytrain[i] * (alphaJold - self.alphas[j])
            self.updateEk(i)          # 更新误差缓存 E1

            # 《统计学习方法》7.115 p148
            b1 = self.b - Ei - self.Ytrain[i] * (self.alphas[i] - alphaIold) * self.K[i, i] \
                 - self.Ytrain[j] * (self.alphas[j] -alphaJold) * self.K[i, j]
            b2 = self.b - Ej - self.Ytrain[i] * (self.alphas[i] - alphaIold) * self.K[i, j] \
                 - self.Ytrain[j] * (self.alphas[j] -alphaJold) * self.K[j, j]

            if (0 < self.alphas[i]) and (self.C > self.alphas[i]):#P148
                self.b = b1
            elif (0 < self.alphas[j]) and (self.C > self.alphas[j]):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def smoP(self):
        # 完整SMO算法外循环，与smoSimple有些类似，但这里的循环退出条件更多一些
        maxIter=self.m * 10
        iter = 0
        entireSet = True
        alphaPairsChanged = 0
        # 至少循环maxIter次 and (存在可以改变的alpha_Pairs or 遍历一遍查找可优化的alpha_piars）
        while (iter < maxIter) and ((alphaPairsChanged > 0) or entireSet):
            alphaPairsChanged = 0
            #  当entireSet=true or 非边界alpha对没有了；就开始寻找 alpha对，然后决定是否要进行else。
            if entireSet:
                # 在数据集上,遍历所有alpha
                for i in range(self.m):
                    alphaPairsChanged += self.innerL(i) # 是否存在alpha对，存在就+1
                iter += 1
            else:
                # 遍历所有的非边界alpha值，0<alpha<C
                nonBoundIs = nonzero((self.alphas > 0) * (self.alphas < self.C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.innerL(i)
                    # print("non-bound iter=%d i=%d, %d pairs changed " % (iter, i, alphaPairsChanged))
                iter += 1
            # 如果找到alpha对，就优化非边界alpha值，否则，就重新进行寻找
            if entireSet:
                entireSet = False  #已经进行了遍历
            elif alphaPairsChanged == 0:  #没有可以优化的alhpa piars 进行全遍历
                entireSet = True
            # print("%d iteration" % iter)
        # if self.kTup[0] == 'lin':
        #     self.calcWs()
        return self.b, self.alphas

    def kernelTrans(self,X,Xi):  # calc the kernel or transform data to a higher dimensional space
        m, n = shape(X)
        X = mat(X)
        Xi=mat(Xi)
        ki = mat(zeros((m, 1)))
        if self.kTup[0] == 'lin':
            # linear kernel:   m*n * n*1 = m*1
            ki = self.X * Xi.T
        elif self.kTup[0] == 'rbf':
            for j in range(m):
                deltaRow = X[j, :] - Xi
                d2 = deltaRow * deltaRow.T
                ki[j] = deltaRow * deltaRow.T
            # 径向基函数的高斯版本
            ki = exp( ki / (-1 * self.kTup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
        else:
            raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
        return ki

    def kernelMatrix(self):  # calc the kernel matrix
        # A = self.X[i, :]  # data[i,:] 数据的i行
        if self.kTup[0] == 'lin':
            for i in range(self.m):
                # linear kernel:   m*n * n*1 = m*1
                self.K[:, i] = self.X * self.X[i, :].T
        elif self.kTup[0] == 'rbf':
            for i in range(self.m):
                # self.K[:, i] = mat(zeros((self.m, 1)))  # 初始化i列 ,m x 1 向量
                for j in range(self.m):
                    deltaRow = self.X[j, :] - self.X[i, :]   # X[j,： ]-X[i,: ]
                    self.K[j,i] = deltaRow * deltaRow.T
                # 径向基函数的高斯版本
                self.K[:, i] = exp(self.K[:, i] / (-1 * self.kTup[1] ** 2))  # KTup[1]=sigma
        else:
            raise NameError('Houston We Have a Problem -- That Kernel is not recognized')

    def calcWs(self):
        self.ws = zeros((self.n, 1))  # 先置零 避免重复累加计算
        for i in range(self.m):
            self.ws += multiply(self.alphas[i] * self.Ytrain[i], self.X[i, :].T)
        print("w1/w0 = %.3f" % float(self.ws[1]/self.ws[0]))

    def Porb_cro_err(self):
        if self.kTup[0] == 'lin':
            if self.ws[0]==0 : self.calcWs()
            # Ytrain_pred = sign(w^T * x + b)
            fx=zeros(shape=(self.m,1))
            sws = self.ws
            for i in range(self.m):
                sxi = self.X[i,:]
                fx[i] = dot(sws.T,sxi.T)+self.b
            self.Ytrain_pred = sign(fx)
        elif self.kTup[0]== 'rbf':
            # fxi = ( Sum(ai*yi*K(xi,x)) + b )
            fx = zeros(shape=(self.m, 1))
            for i in range(self.m):
                fx[i] = self.K[:,i].T * multiply(self.Ytrain,self.alphas) + self.b
            self.Ytrain_pred = sign(fx)
        else:
            raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
        errY_index = nonzero(self.Ytrain_pred - self.Ytrain)[0]  # 错误标签下标
        croY_index = nonzero(self.Ytrain_pred + self.Ytrain)[0]  # 正确标签下标
        crop = len(croY_index) / self.m
        print("prob_cro= %.2f prob_err=%.2f" % (crop, 1-crop))
        print("%d errYs=" %(len(errY_index)), errY_index)
        print("------------------------------------------")
        return crop, 1 - crop

    def label_predict(self,Xin):
        fx = zeros(len(Xin))
        if self.kTup[0] == 'lin':
            # Ytrain_pred = sign(w^T * x + b)
            if self.ws[0]==0 : self.calcWs()
            sws = self.ws
            for i in range(len(Xin)):
                sxi = Xin[i,:]
                fx[i] = dot(sws.T,sxi.T)+self.b
        elif self.kTup[0] == 'rbf':
            # fxi = ( Sum(ai*yi*K(xi,x)) + b )
            for i in range(len(Xin)):
                ki = self.kernelTrans(self.X, Xin[i, :])
                # ya = multiply(self.Ytrain, self.alphas)
                fx[i] = (ki.T) * multiply(self.Ytrain, self.alphas) + self.b
        #######pred_label ,Fx
        return sign(fx), fx

    def predict_X2(self,X2,Y2):
        plt.xlabel(u"x1")
        plt.xlim(X2.min()-0.1, X2.max()*1.1)
        plt.ylabel(u"x2")
        Y2p, fx2 = self.label_predict(X2)
        errY_index = nonzero(Y2 - Y2p)[0]  # 错误标签下标
        croY_index = nonzero(Y2 + Y2p)[0]  # 正确标签下标
        for i in croY_index:
            if Y2[i] > 0:
                plt.plot(X2[i, 0], X2[i, 1], 'xr')
            else:
                plt.plot(X2[i, 0], X2[i, 1], '*g')
        for i in errY_index:
            if Y2[i] > 0:
                plt.plot(X2[i, 0], X2[i, 1], '^y')
            else:
                plt.plot(X2[i, 0], X2[i, 1], 'vb')
        plt.show()
        crop = len(croY_index) / len(X2)
        print("-------------X2-predict--------------------")
        print("prob_cro= %.2f prob_err=%.2f" % (crop, 1-crop))
        print("%d errYs=" %(len(errY_index)), errY_index)
        print("------------------------------------------")

    def plotfig_svm(self,title,Xtest=[],Ytest=[]):
        # plt.figure(subplt)
        # plt.subplots(subplt)
        plt.xlabel("X")
        plt.xlim(self.X.min() - 0.1, self.X.max() * 1.1)
        plt.ylabel("Y")
        plt.title(title+" C:"+str(self.C)+" tol:"+str(self.tol))
        if len(Ytest) != 0:
            #预测所给Test标签
            Ytest_pre= self.label_predict(Xtest)[0]
            ###获取对应点的下标
            errYtest_index = nonzero(Ytest - Ytest_pre)[0]  # 错误标签下标
            croYtest_index = nonzero(Ytest + Ytest_pre)[0]  # 正确标签下标
            ###画出 测试集的具体表现

            ### 不同类型用不同标记表示
            cro_pos = [i for i in croYtest_index if Ytest_pre[i] > 0] #正确的+标签
            cro_neg = [i for i in croYtest_index if Ytest_pre[i]< 0]  #正确的-标签
            plt.scatter(Xtest[cro_pos, 0], Xtest[cro_pos, 1], marker='x', s=40, c='r')
            plt.scatter(Xtest[cro_neg, 0], Xtest[cro_neg, 1], marker='+', s=40, c='g')

            err_pos = [i for i in errYtest_index if Ytest_pre[i] > 0] #错误的+标签
            err_neg = [i for i in errYtest_index if Ytest_pre[i]< 0]  #错误的-标签
            plt.scatter(Xtest[err_pos, 0], Xtest[err_pos, 1], marker='^', s=35, c='m')
            plt.scatter(Xtest[err_neg, 0], Xtest[err_neg, 1], marker='v', s=35, c='c')

        errY_index = nonzero(self.Ytrain_pred - self.Ytrain)[0]  # 错误标签下标
        croY_index = nonzero(self.Ytrain_pred + self.Ytrain)[0]  # 正确标签下标
        for i in croY_index:
            if self.Ytrain[i] > 0:
                plt.scatter(self.X[i,0], self.X[i,1], marker='x', s=25, c='m')
            else:
                plt.scatter(self.X[i,0], self.X[i,1], marker='*', s=25, c='c')

        for i in errY_index:
            if self.Ytrain[i] > 0:
                plt.scatter(self.X[i,0], self.X[i,1], marker='^', s=30, c='m')
            else:
                plt.scatter(self.X[i,0], self.X[i,1], marker='v', s=30, c='c')

        ###如果数据是二维的画出 数据点和对应的SVR
        if self.ws.shape[0] <= 2 and self.kTup[0] == 'lin':
            w = float(- self.ws[0] / self.ws[1])
            b = float(- self.b / self.ws[1])
            r = float(1 / self.ws[1])
            lp_x1 = list([self.X.min()-0.1, self.X.max()*1.1])
            lp_x2 = []
            lp_x2up = []
            lp_x2down = []
            for x1 in lp_x1:
                lp_x2.append(w * x1 + b)
                lp_x2up.append(w * x1 + b + r)
                lp_x2down.append(w * x1 + b - r)
            lp_x2 = list(lp_x2)
            lp_x2up = list(lp_x2up)
            lp_x2down = list(lp_x2down)
            plt.plot(lp_x1, lp_x2, 'b')
            plt.plot(lp_x1, lp_x2up, 'b--')
            plt.plot(lp_x1, lp_x2down, 'b--')
        elif self.kTup[0] == 'rbf':
            print("how to draw the margin of rbf kernel ?")
        print("******************END**********************")
        # plt.show()

    #功能：画利用核函数进行分类的图
    def plotfig_kernel(self,errlabels):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(self.X)):
            if(self.X[i]==1.0):
                ax.plot(self.X[i][0],self.X[i][1],'r^') #预测标签为-1的标为红色三角形
            else:
                ax.plot(self.X[i][0],self.X[i][1],'bs') #预测标签为1的标为蓝色正方形
        for j in errlabels:
            ax.plot(self.X[j][0],self.X[j][1],'og') #预测标签为1的标为蓝色正方形
        plt.show()

def TSVMtrain(X1,Y1,X2,Cu,Cl,Csvm,Svmtoler,kernel):
    # X1, Y1, X2, Cu, Cl, Csvm, kernel
    print("-----------------TSVM-START---------------------")
    tsvm=SVM(X1, Y1, C=Csvm, toler=Svmtoler, kTup=kernel)
    tsvm.smoP()  # 基于带标签样本 先训练一个SVM
    tsvm.calcWs()
    Ytrain = Y1
    Y2_label,fx2 = tsvm.label_predict(X2)  # label , fx = w^Tx + b

    X_all = concatenate((X1, X2), axis = 0)                   #混合数据
    Y_all = concatenate((Ytrain, Y2_label), axis = 0)  #混合标签

    posY2_index = argwhere(Y2_label == 1)    # +标签下标
    negY2_index = argwhere(Y2_label == -1)   # -标签下标
    posY_len = len(posY2_index)
    negY_len = len(negY2_index)
    #根据正负样本比例调整对应正负样本的C 惩罚值
    Cu_pos = Cu*negY_len/posY_len
    Cu_neg = Cu
    # 样本权重初始化
    X2_weight = ones(len(X2))
    X2_weight[posY2_index] = Cu_pos
    X2_weight[negY2_index] = Cu_neg
    # X1_weight = ones(len(X1))
    while (Cu_pos < Cl) or (Cu_neg < Cl):
        tsvm=SVM(X_all, Y_all, C=Csvm, toler=Svmtoler, kTup=kernel)
        tsvm.smoP() # 基于带标签样本 先训练一个SVM
        tsvm.calcWs()
        while True:
            Y2_label,fx2 = tsvm.label_predict(X2)        # label , fx = w^Tx + b
            epsilon = zeros(len(X2))
            eps_pos = []
            eps_neg = []
            epi=0
            epj=0
            for i in range(len(X2)):
                epsilon[i] = max((0, 1 - Y2_label[i] * fx2[i])) # epsilon = 1-Yi*Fxi > 0  是被错误分类的样本
                if epsilon[i]>0:     # if 1-Yi*Fxi > 0  是被错误分类的样本
                    if Y2_label[i]>0:
                        eps_pos.append([epsilon[i],i])
                    else:
                        eps_neg.append([epsilon[i],i])
            if len(eps_pos)==0 or len(eps_neg)==0:
                print("eps+ = %d,eps- = %d"%(len(eps_pos),len(eps_neg)))
            elif len(eps_pos)==0 + len(eps_neg)==0:
                print("break")
                break
            else:
                eps_pos = array(eps_pos)
                eps_neg = array(eps_neg)
                # 找到+-两类中错误得最多的两个样本以及样本的下标
                ida = argmax(eps_pos[:,0])
                epi = eps_pos[ida,0]
                positive_max_id = eps_pos[ida,1]
                idb = argmax(eps_neg[:, 0])
                epj = eps_neg[idb,0]
                negative_max_id = eps_neg[idb,1]
            #####如果找到这样满足条件数据对，交换他们的label和对应的 C值 并重新训练
            if epi > 0 and epj > 0 and epi + epj > 2.0:
                ###交换标签
                Y2_label[positive_max_id] = Y2_label[positive_max_id] * -1
                Y2_label[negative_max_id] = Y2_label[negative_max_id] * -1
                ###交换对应样本Cu值
                X2_weight[positive_max_id] = Cu_neg
                X2_weight[negative_max_id] = Cu_pos
                X_all = concatenate((X1, X2), axis=0)  # 混合数据
                Y_all = concatenate((Ytrain, Y2_label), axis=0)  # 混合标签
                #对混合数据重新进行训练
                tsvm=SVM(X_all, Y_all, C=Csvm, toler=Svmtoler, kTup=kernel)
                tsvm.smoP()
                tsvm.calcWs()
            else:
                break
        #####增大每一类 无标记样本对应的 C值（权重）
        Cu_pos = min([Cu_pos*2, Cl])
        Cu_neg = min([Cu_neg*2, Cl])
        X2_weight[posY2_index] = Cu_pos
        X2_weight[negY2_index] = Cu_neg
    # tsvm.Porb_cro_err()
    # tsvm.plotfig_svm()
    return tsvm

def TSVMtrain1(svmstruct,X2,Cu,Cl):
    # svmstruct = SVM
    # X1, Y1, X2, Cu, Cl, Csvm, kernel
    # tsvm=SVM(X1, Y1, C=Csvm, toler=0.001, kTup=kernel)
    tsvm = svmstruct  ##外部传入SVM函数结构体
    tsvm.smoP()  # 基于带标签样本 先训练一个SVM
    tsvm.calcWs()
    Ytrain = Y1
    Y2_label,fx2 = tsvm.label_predict(X2)  # label , fx = w^Tx + b

    X_all = concatenate((X1, X2), axis = 0)                   #混合数据
    Y_all = concatenate((Ytrain, Y2_label), axis = 0)  #混合标签

    posY2_index = argwhere(Y2_label == 1)    # +标签下标
    negY2_index = argwhere(Y2_label == -1)   # -标签下标
    posY_len = len(posY2_index)
    negY_len = len(negY2_index)
    #根据正负样本比例调整对应正负样本的C 惩罚值
    Cu_pos = Cu*negY_len/posY_len
    Cu_neg = Cu
    # 样本权重初始化
    X2_weight = ones(len(X2))
    X2_weight[posY2_index] = Cu_pos
    X2_weight[negY2_index] = Cu_neg
    # X1_weight = ones(len(X1))
    while (Cu_pos < Cl) or (Cu_neg < Cl):
        # tsvm=SVM(X_all, Y_all, C=Csvm, toler=0.001, kTup=kernel)
        tsvm.X = mat(X_all)
        tsvm.Ytrain = reshape(Y_all,(len(Y_all),1))

        tsvm.smoP() # 基于带标签样本 先训练一个SVM
        tsvm.calcWs()
        while True:
            Y2_label,fx2 = tsvm.label_predict(X2)        # label , fx = w^Tx + b
            epsilon = zeros(len(X2))
            eps_pos = []
            eps_neg = []
            epi=0
            epj=0
            for i in range(len(X2)):
                epsilon[i] = max((0, 1 - Y2_label[i] * fx2[i])) # epsilon = 1-Yi*Fxi > 0  是被错误分类的样本
                if epsilon[i]>0:     # if 1-Yi*Fxi > 0  是被错误分类的样本
                    if Y2_label[i]>0:
                        eps_pos.append([epsilon[i],i])
                    else:
                        eps_neg.append([epsilon[i],i])
            if len(eps_pos)==0 or len(eps_neg)==0:
                print("eps+ = %d,eps- = %d"%(len(eps_pos),len(eps_neg)))
            elif len(eps_pos)==0 + len(eps_neg)==0:
                print("break")
                break
            else:
                eps_pos = array(eps_pos)
                eps_neg = array(eps_neg)
                # 找到+-两类中错误得最多的两个样本以及样本的下标
                ida = argmax(eps_pos[:,0])
                epi = eps_pos[ida,0]
                positive_max_id = eps_pos[ida,1]
                idb = argmax(eps_neg[:, 0])
                epj = eps_neg[idb,0]
                negative_max_id = eps_neg[idb,1]
            #####如果找到这样满足条件数据对，交换他们的label和对应的 C值 并重新训练
            if epi > 0 and epj > 0 and epi + epj > 2.0:
                ###交换标签
                Y2_label[positive_max_id] = Y2_label[positive_max_id] * -1
                Y2_label[negative_max_id] = Y2_label[negative_max_id] * -1
                ###交换对应样本Cu值
                X2_weight[positive_max_id] = Cu_neg
                X2_weight[negative_max_id] = Cu_pos
                X_all = concatenate((X1, X2), axis=0)  # 混合数据
                Y_all = concatenate((Ytrain, Y2_label), axis=0)  # 混合标签
                #对混合数据重新进行训练
                # tsvm=SVM(X_all, Y_all, C=Csvm, toler=0.001, kTup=kernel)
                tsvm.X = mat(X_all)
                tsvm.Ytrain = reshape(Y_all,(len(Y_all),1))
                tsvm.smoP()
                tsvm.calcWs()
            else:
                break
        #####增大每一类 无标记样本对应的 C值（权重）
        Cu_pos = min([Cu_pos*2, Cl])
        Cu_neg = min([Cu_neg*2, Cl])
        X2_weight[posY2_index] = Cu_pos
        X2_weight[negY2_index] = Cu_neg
    tsvm.Porb_cro_err()
    tsvm.plotfig_svm()

def loadShuffleData(fileName):
    dataMat = []; Ytrain = []
    fr = loadtxt(fileName, dtype=float, delimiter='\t')
    # print(fr.shape)
    dataMat = fr[:,0:-1]
    Ytrain = fr[:,-1]
    ###打乱数据及标签
    seed = random.randint(0,1000)  ####392 ,692,87,439,793，743
    print(seed)
    random.seed(seed)
    index = list(range(len(dataMat)))
    random.shuffle(index)
    return dataMat[index],Ytrain[index]

def data_split(data,label,prop=[3,3,4]):
    ##打乱数据
    # index = list(range(len(data)))
    # random.shuffle(index)
    # data = data[index]
    # label = label[index]
    #对数据进行归一化
    for j in range(size(data, axis=1)):
        data[:, j] = (data[:, j] - min(data[:, j])) / (max(data[:, j]) - min(data[:, j]))
    #分割数据
    prop_arr = array([prop[0],prop[0]+prop[1]])
    sp = prop_arr * len(data)
    splitp_arr = [int(i) for i in around(sp)] #按下标 划分数据

    data1 = data[0:splitp_arr[0]]
    data2 = data[splitp_arr[0]:splitp_arr[1]]
    data3 = data[splitp_arr[1]:]

    label1 = label[0:splitp_arr[0]]
    label2 = label[splitp_arr[0]:splitp_arr[1]]
    label3 = label[splitp_arr[1]:]
    # print(data1.shape,label1.shape)
    return data1,label1,data2,label2,data3,label3

# def testRbf(k1=1.3):
    # flname = 'rbftrdata.txt'
    # dataArr, labelArr = loadShuffleData(flname)
    # svmrbf = SVM(dataArr, labelArr, 200, 0.0001, ('rbf', k1))
    # svmrbf.smoP() # C=200 important
    # svmrbf.Porb_cro_err()
    # svmrbf.plotfig_svm()
    # X2, Y2 = loadShuffleData('rbftestdata.txt')
    # svmrbf.predict_X2(X2,Y2)
    #
    # X1,Y1,X2,Y2 = data_split(dataArr,labelArr,0.6)
    # TSVMtrain(X1, Y1, X2, Cu=0.0001, Cl=1.9, Csvm=200,kernel=('rbf',k1))

if __name__ == "__main__":
    # 无核函数的测试
    flname = 'iris14.txt'
    # flname = 'svmtest.txt'
    data1, label1 = loadShuffleData(flname)
    X1,Y1,X2,Y2,X3,Y3 = data_split(data1,label1,prop=[0.1,0.6,0.3])
    Ct = 6
    tol = 0.0001
####仅仅依靠SVM
    svm1 = SVM(X1, Y1, C=Ct, toler=tol, kTup=('lin',0))
    svm1.smoP()
    svm1.Porb_cro_err()
    XX = concatenate((X2,X3),axis=0)
    YY = concatenate((Y2,Y3),axis=0)
    plt.figure(0)
    svm1.plotfig_svm('SVM',XX,YY)
####TSVM
    tsvm = TSVMtrain(X1,Y1,X2,Cu=0.00001,Cl=1.5,Csvm=Ct,Svmtoler=tol,kernel=('lin',0))
    tsvm.Porb_cro_err()
    plt.figure(1)
    tsvm.plotfig_svm('TSVM',X3,Y3)
    plt.show()






