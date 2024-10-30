import { Component, EventEmitter, Input, Output, ViewChild } from '@angular/core';
import { NgbActiveModal } from '@ng-bootstrap/ng-bootstrap';

@Component({
  selector: 'app-video-widget',
  templateUrl: './video-widget.component.html',
  styleUrls: ['./video-widget.component.scss']
})
export class VideoWidgetComponent {
  @ViewChild('citationvideo',{static: false}) citationvideo!: HTMLElement
  @Input() item!: any;
  notplayable = false
  loader = false
  lazyContainer: any
  @Output() refreshChatwindow = new EventEmitter<any>();
  vidURL:  any
  videoElement: any
  constructor(public activeModal: NgbActiveModal){}


  ngOnInit(): void { 
    this.lazyContainer = document.querySelector("div#lazy_container");
    this.videoDynamicRendering()
  }

  videoDynamicRendering(){
    console.log("called")

    this.videoElement = document.querySelector("video");
    // this.vidURL = this.item.newdata ? this.item.url: "https://azureexponentiaai.sharepoint.com/sites/OneTap-Development/_layouts/15/download.aspx?UniqueId=d9886ee9-805a-4355-baee-2abeb1120c19&Translate=false&tempauth=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvYXp1cmVleHBvbmVudGlhYWkuc2hhcmVwb2ludC5jb21AZmIzM2U3ZTEtNmE5OC00ZDVhLWJkMjUtZjQ3YWNmOTUwNzhhIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTcwNDQzNjM4OSIsImV4cCI6IjE3MDQ0Mzk5ODkiLCJlbmRwb2ludHVybCI6Ik5zUTdpeGkxeTdzSXVadm5RUi80dlVDOU5uSGtBYzJSZ1drOTA0amdsaGc9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxNTMiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsImNpZCI6InBjZHJsdmNSZWt5ZmtkNFNXKzc2UUE9PSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJObVl4TmpFMU5qY3RaVE16TXkwMFpqSTNMV0ptTnpVdFpqRmlZak0xTXpSbVlqRTQiLCJhcHBfZGlzcGxheW5hbWUiOiJFeHRlcm5hbEFjY2Vzc1NQQXBwIiwibmFtZWlkIjoiNDRiYmVjNDQtZWM3MC00NmRjLWE2MTUtOTdiODgyMGIyY2FiQGZiMzNlN2UxLTZhOTgtNGQ1YS1iZDI1LWY0N2FjZjk1MDc4YSIsInJvbGVzIjoic2VsZWN0ZWRzaXRlcyBhbGxzaXRlcy5mdWxsY29udHJvbCBhbGxwcm9maWxlcy5yZWFkIiwidHQiOiIxIiwiaXBhZGRyIjoiMjAuMTkwLjE0NS4xNzEifQ.g0rpMn9rP739P2O6XT4pwaxE_WjDenHeRqobZcRwXAw&ApiVersion=2.0"; //dynamic video URL
    this.vidURL = this.item?.url
    this.videoElement.load()
    this.videoElement.addEventListener('error', (event: any) => {
      // console.log("error")
      if(event.type == 'error'){
        // console.log("inside error")
        this.notplayable = true
        this.videoElement.remove()
      }
    });

    
    this.videoElement.onloadstart = ()=>{
      //video responsive width and height
      let width = this.videoElement.parentNode?.parentElement?.clientWidth as number - 50
      let height = this.videoElement.parentNode?.parentElement?.clientHeight as number

      this.videoElement.width = width;
      this.videoElement.height = window.innerHeight as number - height
      // console.log("Video has started loading successfully!");

      this.videoElement.currentTime = this.item.duration;
      this.videoElement.controls = true;
      
    } 

    this.videoElement.addEventListener("loadeddata", () => {
      // console.log("Video has loaded successfully!");
      this.videoElement.play()
    });
  
  }

  refreshChat(){
    this.loader = true
    this.refreshChatwindow.emit(this.item)
  }

  closeModal(){
    this.videoElement.remove()
    this.activeModal.close()
  }
}
