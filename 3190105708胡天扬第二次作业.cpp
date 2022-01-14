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
	printf("���������ĸ�����");
	scanf("%d", &m );
	printf("��������ݣ�");
	
	head = (Node*)malloc( sizeof(Node) );     //�������� 
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
	
	head = DeleteRepeat(head, m);        //ɾ������ֵ��ͬ�Ľ�� 
	p = head->next;                      //������� 
    while(p)
	{       
		printf( "%d ", p->data );
		p = p->next;
	}
	printf( "\n" );
	return 0;
}


Node* DeleteRepeat( Node *q, int n ){
	int NewData[Max];         //����������бȽ� 
	int i = 0;
	Node *p1, *p2;
	p1 = q->next; 
	p2 = p1->next;
	NewData[0] = abs( p1->data );     //�Ȱ������һ������������ 
	while( p2 != NULL ){
		int j, flag = 1;
		for( j=0; j<=i; j++ ){        //�������� 
			if( NewData[j] == abs( p2->data ) ){     //�����������ҵ�����ֵ��ͬ������ɾ���ڵ�
				p1->next = p2->next;
				free( p2 );
				p2 = p1->next;
				i++; 
				flag = 0;          
				break;          
			}
		}
		if( flag ){               //��û���ҵ���������ݼ������� 
			NewData[++i] = abs( p2->data );			
			p1 = p2;
			p2 = p1->next;
		}
	}
	return q;
}
