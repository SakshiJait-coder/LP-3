#include<stdio.h>
long long int power(int a, int b, int mod)
{
long long int t;
if(b==1)
return a;
t=power(a,b/2,mod);
if(b%2==0)
return (t*t)%mod;
else
return (((t*t)%mod)*a)%mod;
}
long int calculateKey(int a, int x, int n)
{
return power(a,x,n);
}
void main()
{
int p,g,x,a,y,b;
printf("Enter the value of p and g : ");
scanf("%d%d",&p,&g);
printf("Enter the value of x for the first person : ");
scanf("%d",&x);
a=power(g,x,p);
printf("Enter the value of y for the second person : ");
scanf("%d",&y);
b=power(g,y,p);
printf("key for the first person is :%lld\n",power(b,x,p));
printf("key for the second person is :%lld\n",power(a,y,p));
}