#include<stdio.h>
#include<stdlib.h>
#define Max 80 

typedef struct Node{
	int data;
	struct Node* next;
}Node; 

Node* DeleteRepeat( Node *q, int n );

int main(void)
{
	int m; 
	Node *p, *head, *tail;
	printf("保存整数的个数：");
	scanf("%d", &m );
	printf("保存的数据：");
	
	head = (Node*)malloc( sizeof(Node) );     //创建链表 
	tail = head;
	int i = 0;
	while( m > 0 && i < m )           
	{          
		p = (Node*)malloc( sizeof(Node) );
		scanf("%d", &p->data);
		tail->next = p;
		tail = p;
		i++;
	}
	tail->next = NULL;
	
	head = DeleteRepeat(head, m);        //删除绝对值相同的结点 
	p = head->next;                      //输出链表 
    while(p)
	{       
		printf( "%d ", p->data );
		p = p->next;
	}
	printf( "\n" );
	return 0;
}


Node* DeleteRepeat( Node *q, int n ){
	int NewData[Max];         //创建数组进行比较 
	int i = 0;
	Node *p1, *p2;
	p1 = q->next; 
	p2 = p1->next;
	NewData[0] = abs( p1->data );     //先把链表第一个数放入数组 
	while( p2 != NULL ){
		int j, flag = 1;
		for( j=0; j<=i; j++ ){        //遍历数组 
			if( NewData[j] == abs( p2->data ) ){     //若在数组中找到绝对值相同的数则删除节点
				p1->next = p2->next;
				free( p2 );
				p2 = p1->next;
				i++; 
				flag = 0;          
				break;          
			}
		}
		if( flag ){               //若没有找到则放入数据继续遍历 
			NewData[++i] = abs( p2->data );			
			p1 = p2;
			p2 = p1->next;
		}
	}
	return q;
}
