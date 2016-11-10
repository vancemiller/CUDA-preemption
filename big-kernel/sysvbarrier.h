/* 
 * use system V ipc primitives to 
 * build barrier synchronization
 */ 

#ifndef _SYS_V_BARRIER_H_
#define _SYS_V_BARRIER_H_

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>

typedef int BARRIER;
typedef int LOCK;

static int barrier_reset(BARRIER barrier, int nprocs) {
  return semctl(barrier, 0, SETVAL, nprocs);
}

static int _get(int _ID) {
  int id = semget(_ID, 1, 0);
  if(id == -1) {
    perror("cannot get semaphore\n");
    exit(1);
  }
  return id;
}

static int _create(int _ID) {
  int sid = semget(_ID, 1, IPC_CREAT | 0666);
  if(sid == -1) {
    perror("semget(create) failed!\n");
    exit(1);
  }
  return sid;
}

static BARRIER barrier_create(int _ID, int nprocs) {
  BARRIER barrier = _create(_ID);
  barrier_reset(barrier, nprocs);
  return barrier;
}

static BARRIER barrier_get(int _ID) {
  return _get(_ID);
}

static LOCK lock_create(int _ID) {
  LOCK lk = _create(_ID);
  semctl(lk, 0, SETVAL, 1);
  return lk;
}

static LOCK lock_get(int _ID) {
  return _get(_ID);
}

static void lock(LOCK lkid) {
  struct sembuf sop;
  sop.sem_num = 0;
  sop.sem_op = -1;
  sop.sem_flg = 0;
  semop(lkid, &sop, 1);
}

static void unlock(LOCK lkid) {
  struct sembuf sop;
  sop.sem_num = 0;
  sop.sem_op = 1;
  sop.sem_flg = 0;
  semop(lkid, &sop, 1);
}

static void barrier_complete(BARRIER barrier, int procid) {
  struct sembuf sop;
  sop.sem_num = 0;
  sop.sem_op = -1;
  sop.sem_flg = 0;
  semop(barrier, &sop, 1);
}

static int _barrier_wait(BARRIER barrier, int procid, int nprocs) {
  struct sembuf sop;
  sop.sem_num = 0;
  sop.sem_op = 0;
  sop.sem_flg = 0;
  semop(barrier, &sop, 1);
}

static void barrier_wait(BARRIER barrier, int procid, int nprocs) {
  barrier_complete(barrier, procid);
  _barrier_wait(barrier, procid, nprocs);
  if(procid == 0)
    barrier_reset(barrier, nprocs);
}

static void barrier_wait_ex(BARRIER barrierX, BARRIER barrierY, int procid, int nprocs) {
  if(procid == 0) {
    barrier_reset(barrierY, nprocs);
  }
  barrier_complete(barrierX, procid);
  _barrier_wait(barrierX, procid, nprocs);
}

#endif

