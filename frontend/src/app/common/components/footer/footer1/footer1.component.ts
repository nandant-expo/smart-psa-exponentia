import { Component } from '@angular/core';

@Component({
  selector: 'app-footer1',
  template: `<div class="d-flex justify-content-center">
      <span class="text-dark">{{ currentYear }}©</span>
      <span class="text-gray">Powered by Exponentia AI</span>
      </div>
      `,
  styles:[
    `.text-dark{
      color:#000;
      font-weight:600;
    }
    .text-gray{
      color: #6e6d6d;
      font-weight: 600;
    }
    .text-gray:hover{
      color:#4F2170;
    }
    `
  ]
})
export class Footer1Component {
  currentYear = new Date().getFullYear();
}
