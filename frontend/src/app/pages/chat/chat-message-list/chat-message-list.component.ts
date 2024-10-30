import { Component, EventEmitter, Input, OnDestroy, OnInit, Output } from '@angular/core';
import { ChatService } from '../services/chat.service';
import { ActivatedRoute } from '@angular/router';
import { HeaderService } from 'src/app/common/services/header.service';
import { OverlayPanel } from 'primeng/overlaypanel';
import { AbstractControl, FormBuilder, FormGroup } from '@angular/forms';
import { MessageService } from 'primeng/api';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-chat-message-list',
  templateUrl: './chat-message-list.component.html',
  styleUrls: ['./chat-message-list.component.scss']
})
export class ChatMessageListComponent implements OnInit, OnDestroy {
  likeFlag = false;
  dislikeFlag = false;
  likeForm: FormGroup = new FormGroup({});
  dislikeForm: FormGroup = new FormGroup({});
  saveFeedback2Subscription = new Subscription();
  stateOptions: any[] = [
    { label: 'Incorrect', value: 'incorrect' },
    { label: 'Partially Correct', value: 'partially_correct' }
  ];

  @Input() messageList!: any;
  @Input() querySent = false;
  @Input() chatFlow: any;
  @Input() lastIndex!: number;
  @Input() currentChatId: any = {};
  @Output() openCitationWindow = new EventEmitter<any>();
  @Output() regenerateQuery = new EventEmitter<any>();
  @Output() queryData = new EventEmitter<any>();

  constructor(
    private activatedRoute: ActivatedRoute,
    private readonly formBuilder: FormBuilder,
    private readonly messageService: MessageService,
    private chatService: ChatService,
    private headerService: HeaderService,
  ) { }
  
  ngOnInit(): void {
    this.toggleExportIcon();
    this.createLikeForm();
    this.createDislikeForm();
  }

  get likeDetailsGetter(): AbstractControl {
    return this.likeForm.get('likeDetails') as AbstractControl
  }
  
  get incorrectPartiallyCorrectGetter(): AbstractControl {
    return this.dislikeForm.get('incorrectPartiallyCorrect') as AbstractControl
  }

  get dislikeDetailsGetter(): AbstractControl {
    return this.dislikeForm.get('dislikeDetails') as AbstractControl
  }

  createLikeForm = () => {
    this.likeForm = this.formBuilder.group({
      likeDetails: [''],
    });
  }

  createDislikeForm = () => {
    this.dislikeForm = this.formBuilder.group({
      incorrectPartiallyCorrect: [''],
      dislikeDetails: [''],
    });
  }

  openCitation(item: any,parentIndex: any){
    item.parentIndex = parentIndex;
    this.openCitationWindow.emit(item);
  }

  retryQuery(){
    this.messageList.pop();
    let query = this.messageList[this.messageList.length - 1].data[0].content;
    this.chatService.query.next(query);
    this.regenerateQuery.emit();
  }
  
  raiseQuery(item: any){
    this.chatService.querySent.next(true);
    this.chatService.query.next(item);
    this.queryData.emit();
  }

  sendFeedback(event: any, index: any, oldIndex: any, type: number, likeDislikeOp: OverlayPanel) {
    if (this.messageList[oldIndex].feedback === 2) {
      // for locking feedback
      likeDislikeOp.toggle(event)
      this.messageList[oldIndex].feedback = type

      if ((this.messageList.length - 1) === index) {
        // if it is last element scroll to bottom
        const messageListEl = document.querySelector('.message-list');
  
        if (messageListEl?.scrollTop) {
          setTimeout(() => {
            // wait 1s
            messageListEl.scrollTop = messageListEl?.scrollHeight
          }, 1000);
        }
      }
    }
  }

  toggleExportIcon = () => {
    this.activatedRoute.params.subscribe(params => {
      const { id, type } = params;

      if (id && type) {
        this.headerService.setIsExportIconVisible(true)
      } else {
        this.headerService.setIsExportIconVisible(false)
      }
    })
  }

  onLikeOpHide = (index: number) => {
    if (!this.likeFlag && this.messageList[index].feedback != 0) {
      // if not submit and not disliked
      this.messageList[index].feedback = 2; // default value
    } else {
      this.likeFlag = false;
    }
    this.likeForm.reset();
  }

  onDislikeOpHide = (index: number) => {
    if (!this.dislikeFlag && this.messageList[index].feedback != 1) {
      // if not submit and not liked
      this.messageList[index].feedback = 2; // default value
    } else {
      this.dislikeFlag = false;
    }
    this.dislikeForm.reset();
  }

  submitLikeForm = (
    event: any,
    index: number,
    type: number,
    thankYouOp: OverlayPanel
  ) => {
    this.likeFlag = true;

    const reqObj = {
      "chat_id": this.currentChatId ?? {},
      "index": index ?? null,
      "feedback": type ?? null,
      "assessment": null,
      "feedback_description": this.likeDetailsGetter?.value ?? null
    }

    this.saveFeedback2Subscription =
      this.chatService.saveFeedBack2(reqObj).subscribe({
        next: () => {
          thankYouOp.show(event);
          this.messageList[index].feedback = type;

          setTimeout(() => {
            // hide after 1.5s
            thankYouOp.hide();
          }, 1500);
        },
        error: (saveFeedBack2Error: any) => {
          this.messageList[index].feedback = 2;
          console.log(
            'Something went wrong while saving feedback. ERROR -> ',
            saveFeedBack2Error
          );
          this.messageService.add({
            severity: 'error',
            summary: 'Error',
            detail: 'Something went wrong while saving feedback.'
          });
        }
      })
  }
  
  submitDislikeForm = (
    event: any,
    index: number,
    type: number,
    thankYouOp: OverlayPanel
  ) => {
    this.dislikeFlag = true;

    const reqObj = {
      "chat_id": this.currentChatId ?? {},
      "index": index ?? null,
      "feedback": type ?? null,
      "assessment": this.incorrectPartiallyCorrectGetter?.value,
      "feedback_description": this.dislikeDetailsGetter?.value ?? null
    }

    this.saveFeedback2Subscription =
      this.chatService.saveFeedBack2(reqObj).subscribe({
        next: () => {
          thankYouOp.show(event);
          this.messageList[index].feedback = type;

          setTimeout(() => {
            // hide after 1.5s
            thankYouOp.hide();
          }, 1500);
        },
        error: (saveFeedBack2Error: any) => {
          this.messageList[index].feedback = 2;
          console.log(
            'Something went wrong while saving feedback. ERROR -> ',
            saveFeedBack2Error
          );
          this.messageService.add({
            severity: 'error',
            summary: 'Error',
            detail: 'Something went wrong while saving feedback.'
          });
        }
      })
  }

  ngOnDestroy(): void {
      this.saveFeedback2Subscription.unsubscribe();
  }
}
