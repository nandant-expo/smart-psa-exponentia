import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { ChatService } from '../services/chat.service';

@Component({
  selector: 'app-chat-history-list',
  templateUrl: './chat-history-list.component.html',
  styleUrls: ['./chat-history-list.component.scss']
})
export class ChatHistoryListComponent implements OnInit {
  loadingArr:any = Array(10).map(() => 25 * Math.random())
  historyList: any = [];
  objId: any;
  querySent: boolean = false;
  stream: boolean = false;
  $newQuery: any;
  $querySent: any;
  $stream: any
  constructor(
    private router: Router,
    private chatService: ChatService,
  ) {
    let decodedparam = this.router.url.split("/");
    this.objId = typeof decodedparam[3] != 'undefined' ? JSON.parse(atob(decodeURIComponent(decodedparam[3]))): null;
    if(this.objId != null){
      this.scrollToActiveMessage();
    }
  }

  ngOnInit(): void {
    this.getHistoryList();
    this.$newQuery = this.chatService.newQuery.subscribe((res: any) => {
      if (res != null) {
        this.historyList[0].records.unshift(res);
        this.objId = res._id;
      }
    })

    this.$querySent = this.chatService.querySent.subscribe((res: boolean)=>{
      this.querySent = res;
    })
    this.$stream  = this.chatService.stream.subscribe((res)=>{
      this.stream = res
    })
  }

  ngOnDestroy(){
    this.$newQuery.unsubscribe()
    this.$querySent.unsubscribe()
    this.$stream.unsubscribe()
  }
  
  scrollToActiveMessage() {
    setTimeout(() => {  
      const element = document.getElementsByClassName("active")[0] as HTMLElement;
      typeof element != 'undefined' ? element.scrollIntoView({ behavior: "smooth", block: "center"}): '';
    }, 600);
  }

  openNewChat() {
    this.objId = null
    this.chatService.chatFlow.next('general');
    this.router.navigate(['/structure-data-analysis']);
  }
  
  editChat(item: any) {
    this.objId = item._id;
    let objId = encodeURI(btoa(JSON.stringify(item._id)).toString())
    let type = encodeURI(btoa(JSON.stringify(item.chat_flow)).toString())
    this.router.navigate(['/structure-data-analysis/c/',objId,type]);
  }

  getHistoryList() {
    this.chatService.getChatMessageHistoryList({}).subscribe((res: any) => {
      this.historyList = res.data;
      if(this.objId != null){
        this.scrollToActiveMessage();
      }
    })
  }

  getActiveMessage(id: any) {
    if(this.objId!=null){ 
      if (Object.values(this.objId)[0] == Object.values(id)[0]) {
        return true
      }else{
        return false
      }
    }else{
      return false
    }
  }

  deleteConversion(id: any,parentIndex: any,index: any){ 
    this.chatService.deleteChatMessageHistory({"chat_id": id }).subscribe((res: any) => {
      this.historyList[parentIndex].records.splice(index, 1);
      this.openNewChat();
    })
  }

}
