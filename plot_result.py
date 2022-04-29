import matplotlib.pyplot as plt

p = [0,0.3,0.5,1.0]
v = [0.9126,0.9105,0.8725,0.8188]
fig, ax = plt.subplots(1)
ax.plot(p,v,color='blue',linewidth=3.0,linestyle='-')
# ax.set_xlim((0,1))
ax.set_ylim((0.8,0.95))
ax.set_xlabel('p')
ax.set_ylabel('F1')
plt.grid()

fea = [0,1,2,3,4]
v1 = [0.6567,0.8378,0.9105,0.9190,0.9345]
v2 = [0.7130,0.8512,0.9121,0.9190,0.9291]

fig2, ax2 = plt.subplots(1)
ax2.plot(fea,v1,color='blue',linewidth=3.0,linestyle='-')
ax2.plot(fea,v2,color='red',linewidth=3.0,linestyle='-')
# ax.set_xlim((0,1))
# ax.set_ylim((0.8,0.95))
ax2.set_xlabel('max_features')
ax2.set_ylabel('F1')
ax2.set_xticks((range(len(fea))))
ax2.set_xticklabels(["10","100","500","1000",">1000"])
plt.grid()
plt.legend(["LinearSVC", "Ensemble"])
plt.show()
