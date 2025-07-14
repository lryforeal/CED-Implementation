#include <bits/stdc++.h>

using namespace std;

const double T=1000;										// Maximum iteration 
const double alpha=0.005;									// Learning rate
const double delta=1.0;										// Direct policy constraint parameter (chosen from (0.5,1.0])
const double epsilon=0.1;									// Indirect policy penalty paramter
const int stage_num=3;
const int action_max=5;
const int action_num[stage_num]={2,2,5};

const double PM[5][5]={										// Payoff matrix at Stage 3 (5-action RPS game)
	0,-1,1,-1,1,
	1,0,-1,-1,1,
	-1,1,0,-1,1,
	1,1,1,0,-1,
	-1,-1,-1,1,0
};

double mu_beta[stage_num][action_max];
double nu_beta[stage_num][action_max];
double l[stage_num][action_max],r[stage_num][action_max];	// [l,r] is the range of direct policy constraint
double Q[stage_num][action_max][action_max];
double R[stage_num][action_max][action_max];
double mu[stage_num][action_max];
double nu[stage_num][action_max];

void init(){
	R[0][0][0]=1,R[0][0][1]=0,R[0][1][0]=-2,R[0][1][1]=4;
	R[1][0][0]=1,R[1][0][1]=0,R[1][1][0]=-2,R[1][1][1]=3;
	for (int i=0;i<5;i++)
		for (int j=0;j<5;j++)
			R[2][i][j]=PM[i][j];
	for (int i=0;i<stage_num;i++){
		for (int j=0;j<action_num[i];j++){
			if (i==2){
				if (j==0) mu_beta[i][j]=0.6;				// Behavior policy: mu=(0.6,0.1,0.1,0.1,0.1) at Stage 3 (i==2)
				else mu_beta[i][j]=0.1;
			}
			else mu_beta[i][j]=1.0/action_num[i];			// Behavior policy: mu=(1/3,1/3,1/3) at Stage 1 and Stage 2
			nu_beta[i][j]=1.0/action_num[i];				// Behavior policy: nu is uniform policy at all 3 stages
			l[i][j]=max(0.0,mu_beta[i][j]-delta);
			r[i][j]=min(1.0,mu_beta[i][j]+delta);
			mu[i][j]=mu_beta[i][j];
			nu[i][j]=nu_beta[i][j];
		}
	}
}

double calc_V(int num){
	double ret=0.0;
	for (int a=0;a<action_num[num];a++)
		for (int b=0;b<action_num[num];b++)
			ret+=mu[num][a]*nu[num][b]*Q[num][a][b];
	return ret;
}

void adjust_mu(int num){									// Applying direct policy constraint by forcing mu to be in the range
	double ok[action_max],s;
	for (int i=0;i<action_num[num];i++)
		ok[i]=true;
	int cnt=action_num[num];
	while (true){											// Iterative adjustment under lower bound l
		s=0.0;
		for (int i=0;i<action_num[num];i++){
			if (ok[i]&&mu[num][i]<=l[num][i]){
				s+=l[num][i]-mu[num][i];
				mu[num][i]=l[num][i];
				ok[i]=false;
				cnt--;
			}
		}
		if (s==0.0) break;
		for (int i=0;i<action_num[num];i++)
			if (ok[i]) mu[num][i]-=s/cnt;
	}
	s=0.0;
	for (int i=0;i<action_num[num];i++)
		if (mu[num][i]>r[num][i]){							// At most one i satisfies the condition since r>0.5
			s=mu[num][i]-r[num][i];
			mu[num][i]=r[num][i];
		}
	for (int i=0;i<action_num[num];i++)
		if (mu[num][i]<r[num][i])
			mu[num][i]+=s/(action_num[num]-1);
}

void update_mu(int num){									// Policy update under direct policy constraint
	double p[action_max],s=0.0;
	for (int i=0;i<action_num[num];i++){
		p[i]=0.0;
		for (int j=0;j<action_num[num];j++)
			p[i]+=nu[num][j]*Q[num][i][j];
		s+=p[i];
	}
	s/=action_num[num];
	for (int i=0;i<action_num[num];i++)
		mu[num][i]+=alpha*(p[i]-s);
	adjust_mu(num);											// Better projection implementation could also be used to replace this simple one
}

void update_nu(int num){									// Policy update under indirect policy penalty
	double p[action_max],s=0.0;
	for (int i=0;i<action_num[num];i++){
		p[i]=0.0;
		for (int j=0;j<action_num[num];j++)
			p[i]-=mu[num][j]*Q[num][j][i];
		p[i]=nu_beta[num][i]*exp(p[i]/epsilon);				// Following Lemma 4.1 in the CED paper
		s+=p[i];
	}
	for (int i=0;i<action_num[num];i++)
		nu[num][i]=p[i]/s;
}

void output(){
	for (int i=0;i<stage_num;i++){
		printf("nu[Stage %d]=(",i+1);
		for (int j=0;j<action_num[i];j++){
			printf("%lf",nu[i][j]);
			if (j!=action_num[i]-1)
				printf(",");
		}
		printf(")\n");
	}
}

int main(){
	init();
	printf("Behavior policy of min-player:\n");
	output();
	printf("\n");
	printf("Applying CED ... \n");
	for (int i=1;i<=T;i++){									// Main loop of CED
		double temp[stage_num];
		for (int j=0;j<stage_num;j++) temp[j]=calc_V(j);
		for (int j=0;j<stage_num;j++)
			for (int a=0;a<action_num[j];a++)
				for (int b=0;b<action_num[j];b++){
					Q[j][a][b]=R[j][a][b];
					if (j>0) continue;						// Only Stage 1 (j==0) is not leaf node
					if (a==b) Q[j][a][b]+=temp[1];
					else Q[j][a][b]+=temp[2];
				}
		for (int j=0;j<stage_num;j++) update_mu(j);
		for (int j=0;j<stage_num;j++) update_nu(j);
	}
	printf("\n");
	printf("Computed equilibrium policy of min-player:\n");
	output();
	return 0;
}
