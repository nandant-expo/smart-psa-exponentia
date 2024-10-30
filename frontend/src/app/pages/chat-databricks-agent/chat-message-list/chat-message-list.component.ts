import { AfterViewInit, Component, EventEmitter, Input, OnChanges, OnInit, Output, SimpleChanges } from '@angular/core';
import { ChatService } from '../services/chat.service';
import { HighlightService } from '../services/highlight.service';

@Component({
  selector: 'app-chat-message-list',
  templateUrl: './chat-message-list.component.html',
  styleUrls: ['./chat-message-list.component.scss']
})
export class ChatMessageListComponent implements OnInit, OnChanges, AfterViewInit{
  @Input() messageList!: any;
  @Input() querySent = false;
  @Input() chatFlow: any;
  @Input() lastIndex!: number;
  @Output() openCitationWindow = new EventEmitter<any>();
  @Output() regenerateQuery = new EventEmitter<any>();
  @Output() queryData = new EventEmitter<any>();
  @Output() saveFeedback = new EventEmitter<any>();

  btnCopyText = 'Copy';
  btnCopyIconName = 'pi pi-copy';
highlighted: boolean = false;
  constructor(private chatService: ChatService,private highlightService: HighlightService ) { }
  
  ngOnInit(): void {
  }

  ngAfterViewInit() {
    setTimeout(() => {
        this.highlightService.highlightAll();
    }, 800);
  }

  ngOnChanges(changes: SimpleChanges): void {
    setTimeout(() => {
        this.highlightService.highlightAll();
    }, 800);
  }
  openCitation(item: any,parentIndex: any){
    item.parentIndex = parentIndex;
    this.openCitationWindow.emit(item);
  }


  copyInputMessage(copyText: any){
    this.btnCopyText = 'Copied';
    this.btnCopyIconName = 'pi pi-check';
    let selBox = document.createElement('textarea');
    selBox.style.visibility = 'none';
    selBox.value = copyText;
    document.body.appendChild(selBox);
    selBox.focus();
    selBox.select();
    document.execCommand('copy');
    document.body.removeChild(selBox);
    setTimeout(() => {
        // change back after 1.5s
        this.btnCopyText = 'Copy';
        this.btnCopyIconName = 'pi pi-copy'
      }, 1000);
  }
  retryQuery(){
    this.messageList.pop();
    let query = this.messageList[this.messageList.length - 1].data.content;
    this.chatService.query.next(query);
    this.regenerateQuery.emit();
  }
  
  raiseQuery(item: any){
    this.chatService.querySent.next(true);
    this.chatService.query.next(item);
    this.queryData.emit();
  }

  sendFeedback(index: any,oldIndex: any,type:number){
    this.messageList[oldIndex].feedback = type
    this.saveFeedback.emit({index:index,type:type});
  }
}
