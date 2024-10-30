import { Component, ElementRef, EventEmitter, Input, OnChanges, OnInit, Output, SimpleChanges, ViewChild } from '@angular/core';
import { ChatService } from '../services/chat.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-chat-ask-a-question',
  templateUrl: './chat-ask-a-question.component.html',
  styleUrls: ['./chat-ask-a-question.component.scss']
})
export class ChatAskAQuestionComponent implements OnInit, OnChanges {
  @ViewChild('messageBox', { read: ElementRef }) messageBox!: ElementRef;
  activeClass = '';
  @Input() id!: any;
  @Output() queryData = new EventEmitter<any>();
  question: string = '';
  emptyError: boolean = false;
  querySent: boolean = false;

  constructor(private chatService: ChatService, private router: Router) {}

  ngOnInit(): void {
    this.activeClass = ''; 
    this.chatService.querySent.subscribe((res: any) => {
      this.querySent = res;
    }) 
  }

  ngOnChanges(changes: SimpleChanges) {
    const dataChange = changes['id'];
    if(dataChange.firstChange === false){
      this.emptyText()
    }
  }

  onValueChange(event: any) {
    event.target.value != '' ? this.activeClass = 'activeicon' : this.activeClass = '';
    if(event.target.value != ''){
      this.messageBox.nativeElement.style.overflow = 'scroll';
      if(this.messageBox.nativeElement.scrollHeight >=130){
        this.messageBox.nativeElement.style.height = '130px';
        this.messageBox.nativeElement.classList.remove('autogrow-box');
      }else{
        this.messageBox.nativeElement.style.height = !this.emptyError ? this.messageBox.nativeElement.scrollHeight + 'px': '24px';
        this.messageBox.nativeElement.classList.add('autogrow-box');
      }
    }else{  
      this.emptyText()
    }
  }

  emptyText() {
    this.question = '';
    this.messageBox.nativeElement.style.height = '24px'
    this.messageBox.nativeElement.style.overflow = 'hidden';
    this.messageBox.nativeElement.classList.remove('autogrow-box');
  }

  emptyErrorFlag(){
    this.emptyError = true;
    setTimeout(() => {
      this.emptyError = false;
    } , 2000);
  }

  onKeyUpChange(event: any) {
    if (event.key === 'Enter' && !event.shiftKey && this.question.replace(/[\n\r\s\t]+/g, '') == '' && !this.querySent) {
      this.emptyErrorFlag()
      this.emptyText()
    }
  }
  onKeyDownChange(event: any) {
    if (event.key === 'Enter' && !event.shiftKey && this.question.replace(/[\n\r\s\t]+/g, '') == '') {
      this.emptyErrorFlag()
      this.emptyText()
    }else{
      if(event.key === 'Enter' && !event.shiftKey && this.question.replace(/[\n\r\s\t]+/g, '') != ''){
        if(!this.querySent){
          this.getChatMessageQueryRes(this.question)
          setTimeout(() => {
            this.emptyText()
          }, 300);
        }
      }
    }
  }
  getChatMessageQuery() {  
    if(this.question.replace(/[\n\r\s\t]+/g, '') != ''){
      if(!this.querySent){
      this.getChatMessageQueryRes(this.question);
      this.emptyText()
      }
    }else{
      this.emptyErrorFlag()
    }
  }
  
  getChatMessageQueryRes(query: any) {
    this.chatService.querySent.next(true);
    this.chatService.query.next(query);
    if(typeof(this.id) != 'undefined'){
      this.queryData.emit();
    } else{
      let objId = encodeURI(btoa(JSON.stringify('new')).toString())
      let type = encodeURI(btoa(JSON.stringify('general')).toString())
      this.router.navigate(['/chat/c/',objId,type]);
    }
  }

}
